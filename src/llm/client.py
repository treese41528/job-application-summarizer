"""
LLM Client for Purdue GenAI Studio.

Wraps the GenAIStudio class to provide:
  - query() / query_json() / query_stream() / query_with_usage()
  - Full RAG lifecycle: upload, KB create/delete, file linking
  - collections= pass-through for RAG-grounded queries
  - Ensemble helpers: run the same prompt across multiple models

Expects genai_studio.py (the GenAIStudio module) to be importable.

GenAI Studio API reference (from genai_studio.py):

    ai = GenAIStudio(validate_model=False)
    ai.select_model("gemma3:12b")

    # Chat
    response = ai.chat(prompt, system=..., temperature=..., collections=[...])
    chat_resp = ai.chat_complete(prompt, ..., collections=[...])
    for chunk in ai.chat_stream(prompt, ..., collections=[...]): ...

    # RAG
    file_info = ai.upload_file("notes.pdf")         -> FileInfo
    kb = ai.create_knowledge_base("name")            -> KnowledgeBase
    ai.add_file_to_knowledge_base(kb.id, file.id)
    ai.remove_file_from_knowledge_base(kb.id, file.id)
    ai.delete_knowledge_base(kb.id)
    ai.list_knowledge_bases()                        -> list[KnowledgeBase]
    ai.list_files()                                  -> list[FileInfo]
    ai.delete_file(file_id)

    # Embeddings
    ai.embed("text")
    ai.similarity("a", "b")
"""

import json
import logging
import re
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ── Lazy import holder ──
_GenAIStudio = None


def _import_genai_studio():
    """
    Import GenAIStudio class, searching project root if needed.
    Returns the GenAIStudio class (not an instance).
    """
    global _GenAIStudio
    if _GenAIStudio is not None:
        return _GenAIStudio

    try:
        from genai_studio import GenAIStudio
        _GenAIStudio = GenAIStudio
        return _GenAIStudio
    except ImportError:
        pass

    # Fallback: add project root to sys.path and retry
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from genai_studio import GenAIStudio
        _GenAIStudio = GenAIStudio
        return _GenAIStudio
    except ImportError:
        raise ImportError(
            "genai_studio.py not found.\n"
            "Place it in the project root directory or ensure it's importable.\n"
            "Also make sure GENAI_STUDIO_API_KEY is set in your environment.\n\n"
            "  cp /path/to/genai_studio.py ./\n"
            "  export GENAI_STUDIO_API_KEY='your-key-here'"
        )


class LLMClient:
    """
    Wrapper around Purdue GenAI Studio for structured LLM interactions.

    Provides:
      - query() / query_json() / query_stream() / query_with_usage()
      - RAG: upload_file, create_kb, add_file_to_kb, delete_kb, etc.
      - collections= on all query methods for RAG-grounded generation
      - ensemble_query() / ensemble_query_json() for multi-model evaluation
      - switch_model() to change model without re-creating the client

    Usage:
        client = LLMClient(model="llama3.1:70b-instruct-q4_K_M")

        # Basic queries
        text = client.query("What is statistics?", temperature=0.3)
        data = client.query_json(prompt, temperature=0.1)

        # RAG-grounded queries
        data = client.query_json(prompt, collections=["kb-uuid-here"])

        # Ensemble (multi-model)
        results = client.ensemble_query_json(
            prompt,
            models=["gemma3:12b", "llama3.1:70b-instruct-q4_K_M"],
            temperature=0.2,
        )
        # results = [{"model": "gemma3:12b", "data": {...}}, ...]
    """

    def __init__(
        self,
        model: str = "llama3.1:70b-instruct-q4_K_M",
        validate_model: bool = False,
        system_prompt: str | None = None,
        timeout: float = 600.0,
    ):
        """
        Initialize the LLM client.

        Args:
            model: GenAI Studio model ID to use.
            validate_model: If True, verify model exists at init time.
            system_prompt: Default system prompt for all queries.
            timeout: HTTP request timeout in seconds (default 600 = 10 min).
        """
        self.model = model
        self._validate_model = validate_model
        self.system_prompt = system_prompt
        self._timeout = timeout
        self._studio = None  # GenAIStudio instance (lazy-initialized)

    # ════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ════════════════════════════════════════════════════════════════════

    def _ensure_initialized(self):
        """Lazy-initialize the GenAI Studio connection on first use."""
        if self._studio is not None:
            return

        GenAIStudio = _import_genai_studio()

        try:
            self._studio = GenAIStudio(
                validate_model=self._validate_model,
                timeout=self._timeout,
            )
        except TypeError:
            # GenAIStudio doesn't accept timeout — init without it
            self._studio = GenAIStudio(validate_model=self._validate_model)
            # Patch timeout on underlying client if possible
            for attr in ("_client", "client"):
                obj = getattr(self._studio, attr, None)
                if obj and hasattr(obj, "timeout"):
                    import httpx
                    obj.timeout = httpx.Timeout(self._timeout)
                    break

        self._studio.select_model(self.model)
        logger.info(f"GenAI Studio client initialized: model={self.model}")

    @property
    def studio(self):
        """Direct access to the underlying GenAIStudio instance."""
        self._ensure_initialized()
        return self._studio

    # ════════════════════════════════════════════════════════════════════
    # CORE QUERY METHODS
    # ════════════════════════════════════════════════════════════════════

    def query(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        max_retries: int = 2,
        collections: list[str] | None = None,
    ) -> str:
        """
        Send a prompt and return the raw text response.

        Args:
            prompt: The user prompt text.
            system: System prompt (falls back to self.system_prompt).
            temperature: Sampling temperature.
            max_tokens: Max response tokens. None = model decides.
            max_retries: Retries on failure.
            collections: KB IDs for RAG-grounded generation.

        Returns:
            Stripped response text from the LLM.
        """
        self._ensure_initialized()

        effective_system = system if system is not None else self.system_prompt

        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if collections:
            kwargs["collections"] = collections

        for attempt in range(max_retries + 1):
            try:
                response = self._studio.chat(
                    prompt,
                    system=effective_system,
                    **kwargs,
                )

                if response and isinstance(response, str):
                    return response.strip()
                elif response:
                    return str(response).strip()
                else:
                    logger.warning(
                        f"Empty LLM response (attempt {attempt + 1}/{max_retries + 1})"
                    )

            except Exception as e:
                logger.warning(
                    f"LLM query failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.0 * (attempt + 1))

        return ""

    def query_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        max_retries: int = 2,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Send a prompt and parse the response as JSON.

        Handles markdown code blocks, trailing commas, preamble text,
        and reasoning-model tags (<think>...</think>).

        Args:
            prompt: The prompt (should instruct LLM to respond in JSON).
            system: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            max_retries: Retries on JSON parse failure.
            collections: KB IDs for RAG-grounded generation.

        Returns:
            Parsed JSON as dict or list.

        Raises:
            ValueError: If JSON cannot be parsed after all retries.
        """
        json_system = system or self.system_prompt or (
            "You are a data extraction assistant. Always respond with valid JSON only. "
            "No markdown formatting, no explanation, no preamble."
        )

        last_error = None
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            raw_response = self.query(
                current_prompt,
                system=json_system,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,
                collections=collections,
            )

            try:
                return self._parse_json_response(raw_response)
            except ValueError as e:
                last_error = e
                logger.warning(
                    f"JSON parse failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    current_prompt = prompt + (
                        "\n\nCRITICAL: Respond with ONLY valid JSON. "
                        "No markdown code fences, no explanation text, "
                        "just the raw JSON object starting with { and ending with }."
                    )

        raise ValueError(
            f"Could not parse LLM response as JSON after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def query_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        collections: list[str] | None = None,
    ) -> Iterator[str]:
        """
        Stream a response, yielding tokens as they arrive.

        Args:
            prompt: The user prompt.
            system: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            collections: KB IDs for RAG-grounded generation.

        Yields:
            Individual token strings as generated.
        """
        self._ensure_initialized()

        effective_system = system if system is not None else self.system_prompt
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if collections:
            kwargs["collections"] = collections

        yield from self._studio.chat_stream(
            prompt,
            system=effective_system,
            **kwargs,
        )

    def query_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        collections: list[str] | None = None,
    ) -> tuple[str, dict]:
        """
        Send a prompt and return both text and token usage stats.

        Args:
            prompt: The user prompt.
            system: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            collections: KB IDs for RAG-grounded generation.

        Returns:
            Tuple of (response_text, usage_dict).
        """
        self._ensure_initialized()

        effective_system = system if system is not None else self.system_prompt
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if collections:
            kwargs["collections"] = collections

        chat_response = self._studio.chat_complete(
            prompt,
            system=effective_system,
            **kwargs,
        )

        return chat_response.content.strip(), chat_response.usage

    # ════════════════════════════════════════════════════════════════════
    # RAG: FILE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════

    def upload_file(self, filepath: str) -> "FileInfo":
        """
        Upload a file to GenAI Studio for RAG indexing.

        Args:
            filepath: Local path to the file.

        Returns:
            FileInfo with server-assigned id and filename.
        """
        self._ensure_initialized()
        logger.info(f"Uploading file: {filepath}")
        info = self._studio.upload_file(filepath)
        logger.info(f"Uploaded: {info.filename} -> {info.id}")
        return info

    def list_files(self) -> list:
        """List all uploaded files on GenAI Studio."""
        self._ensure_initialized()
        return self._studio.list_files()

    def delete_file(self, file_id: str) -> bool:
        """Delete an uploaded file by ID."""
        self._ensure_initialized()
        logger.info(f"Deleting file: {file_id}")
        return self._studio.delete_file(file_id)

    # ════════════════════════════════════════════════════════════════════
    # RAG: KNOWLEDGE BASE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════

    def create_knowledge_base(
        self, name: str, description: str = ""
    ) -> "KnowledgeBase":
        """
        Create a new knowledge base (RAG collection).

        Args:
            name: Human-readable name for the KB.
            description: Optional description.

        Returns:
            KnowledgeBase with server-assigned id.
        """
        self._ensure_initialized()
        logger.info(f"Creating knowledge base: {name}")
        kb = self._studio.create_knowledge_base(name, description)
        logger.info(f"Created KB: {kb.name} -> {kb.id}")
        return kb

    def list_knowledge_bases(self) -> list:
        """List all knowledge bases."""
        self._ensure_initialized()
        return self._studio.list_knowledge_bases()

    def get_knowledge_base(self, kb_id: str) -> "KnowledgeBase":
        """Get details for a specific knowledge base."""
        self._ensure_initialized()
        return self._studio.get_knowledge_base(kb_id)

    def add_file_to_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """Link an uploaded file to a KB for indexing."""
        self._ensure_initialized()
        logger.info(f"Linking file {file_id} -> KB {kb_id}")
        return self._studio.add_file_to_knowledge_base(kb_id, file_id)

    def remove_file_from_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """Unlink a file from a KB."""
        self._ensure_initialized()
        logger.info(f"Unlinking file {file_id} from KB {kb_id}")
        return self._studio.remove_file_from_knowledge_base(kb_id, file_id)

    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base (files are NOT deleted)."""
        self._ensure_initialized()
        logger.info(f"Deleting knowledge base: {kb_id}")
        return self._studio.delete_knowledge_base(kb_id)

    # ════════════════════════════════════════════════════════════════════
    # ENSEMBLE QUERIES
    # ════════════════════════════════════════════════════════════════════

    def ensemble_query(
        self,
        prompt: str,
        models: list[str],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = 1,
        collections: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the same prompt across multiple models sequentially.

        Returns a list of results, one per model:
            [
                {"model": "gemma3:12b", "response": "...", "error": None},
                {"model": "llama3.1:70b-...", "response": "...", "error": None},
            ]

        Failed models are included with error= set and response="".

        Args:
            prompt: The prompt to send to each model.
            models: List of model IDs to query.
            system: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            max_retries: Retries per model.
            collections: KB IDs for RAG-grounded generation.

        Returns:
            List of dicts with model, response, and error keys.
        """
        original_model = self.model
        results = []

        for model_id in models:
            logger.info(f"Ensemble query: switching to {model_id}")
            try:
                self.switch_model(model_id)
                response = self.query(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    collections=collections,
                )
                results.append({
                    "model": model_id,
                    "response": response,
                    "error": None,
                })
            except Exception as e:
                logger.error(f"Ensemble query failed for {model_id}: {e}")
                results.append({
                    "model": model_id,
                    "response": "",
                    "error": str(e),
                })

        # Restore original model
        try:
            self.switch_model(original_model)
        except Exception:
            logger.warning(f"Could not restore model to {original_model}")

        return results

    def ensemble_query_json(
        self,
        prompt: str,
        models: list[str],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = 2,
        collections: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the same JSON prompt across multiple models.

        Returns:
            [
                {"model": "gemma3:12b", "data": {...}, "error": None},
                {"model": "llama3.1:70b-...", "data": {...}, "error": None},
            ]

        Failed models have data={} and error set.
        """
        original_model = self.model
        results = []

        for model_id in models:
            logger.info(f"Ensemble JSON query: switching to {model_id}")
            try:
                self.switch_model(model_id)
                data = self.query_json(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    collections=collections,
                )
                results.append({
                    "model": model_id,
                    "data": data,
                    "error": None,
                })
            except Exception as e:
                logger.error(f"Ensemble JSON query failed for {model_id}: {e}")
                results.append({
                    "model": model_id,
                    "data": {},
                    "error": str(e),
                })

        # Restore original model
        try:
            self.switch_model(original_model)
        except Exception:
            logger.warning(f"Could not restore model to {original_model}")

        return results

    def synthesize(
        self,
        synthesis_prompt: str,
        synthesizer_model: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Switch to the synthesizer model and run a JSON synthesis query.

        Designed to be called after ensemble_query_json() — you format
        the ensemble results into the synthesis_prompt, and this method
        gets the final merged evaluation.

        Args:
            synthesis_prompt: Prompt containing all ensemble results.
            synthesizer_model: Model to use for synthesis (e.g. deepseek-r1:70b).
            system: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            collections: KB IDs for RAG context during synthesis.

        Returns:
            Parsed JSON dict from the synthesizer.
        """
        original_model = self.model
        try:
            self.switch_model(synthesizer_model)
            return self.query_json(
                synthesis_prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                collections=collections,
            )
        finally:
            try:
                self.switch_model(original_model)
            except Exception:
                logger.warning(f"Could not restore model to {original_model}")

    # ════════════════════════════════════════════════════════════════════
    # MODEL MANAGEMENT
    # ════════════════════════════════════════════════════════════════════

    def switch_model(self, model: str):
        """
        Switch to a different LLM model.

        Args:
            model: New model ID.
        """
        old_model = self.model
        self.model = model

        if self._studio is not None:
            self._studio.select_model(model)

        if old_model != model:
            logger.info(f"Model switched: {old_model} → {model}")

    def list_models(self) -> list[str]:
        """List available models on GenAI Studio."""
        self._ensure_initialized()
        return self._studio.models

    # ════════════════════════════════════════════════════════════════════
    # JSON PARSING
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any] | list:
        """
        Parse JSON from an LLM response, handling common formatting issues.

        Tries in order:
        1. Direct JSON parse
        2. Extract from ```json ... ``` code blocks
        3. Extract from ``` ... ``` code blocks
        4. Find first { ... } or [ ... ] in response
        5. Fix trailing commas and retry
        6. Fix single quotes and retry

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed JSON object (dict or list).

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        if not response or not response.strip():
            raise ValueError("Empty response")

        response = response.strip()

        # Strip reasoning model tags (e.g., deepseek-r1 <think>...</think>)
        response = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL
        ).strip()

        # 1. Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code blocks
        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # 3. Find JSON object or array boundaries
        for open_char, close_char in [("{", "}"), ("[", "]")]:
            start = response.find(open_char)
            end = response.rfind(close_char)
            if start != -1 and end > start:
                candidate = response[start : end + 1]

                # Try as-is
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

                # Fix trailing commas: ,} → } and ,] → ]
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

                # Fix single quotes → double quotes
                fixed2 = fixed.replace("'", '"')
                try:
                    return json.loads(fixed2)
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not extract valid JSON from response: {response[:300]}..."
        )