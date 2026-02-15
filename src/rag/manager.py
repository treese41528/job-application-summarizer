"""
Per-Applicant Knowledge Base Manager.

Handles the full RAG lifecycle for each applicant:
  1. Upload their documents (PDFs, DOCXs) to GenAI Studio
  2. Create a dedicated knowledge base
  3. Link uploaded files to the KB
  4. Wait for server-side indexing
  5. Persist metadata so KBs can be reused (for chat, re-evaluation)
  6. Cleanup KBs and files when no longer needed

Metadata is stored as `rag_metadata.json` in each applicant's results
directory, enabling the chat interface to look up existing KBs without
re-uploading.

Usage:
    from src.rag import ApplicantKBManager
    from src.llm.client import LLMClient

    client = LLMClient(model="gemma3:12b")
    manager = ApplicantKBManager(client)

    # Create KB for an applicant
    kb_id = manager.create_applicant_kb(
        applicant_name="Alice Smith",
        folder_name="Alice_Smith",
        source_folder=Path("../vap-search-2025/Alice_Smith/"),
        results_path=Path("data/results/Alice_Smith/"),
    )

    # Use in queries
    data = client.query_json(prompt, collections=[kb_id])

    # Later: get existing KB without re-uploading
    kb_id = manager.get_kb_id("Alice_Smith", results_path)

    # Cleanup
    manager.cleanup_applicant_kb("Alice_Smith", results_path)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

# File extensions we can upload for RAG indexing
UPLOADABLE_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".docx", ".doc",
    ".json", ".py", ".r", ".rtf",
}

# How long to wait after linking files before querying (seconds)
DEFAULT_INDEX_WAIT = 12

# How long to wait between upload and linking (seconds)
DEFAULT_UPLOAD_SETTLE = 2

METADATA_FILENAME = "rag_metadata.json"


@dataclass
class RAGMetadata:
    """
    Persistent metadata for an applicant's RAG resources.

    Stored as rag_metadata.json in the applicant's results directory.
    Allows re-use of existing KBs across processing runs and by the
    chat interface.
    """
    applicant_name: str
    folder_name: str
    kb_id: str
    kb_name: str
    file_ids: list[dict] = field(default_factory=list)
    # Each entry: {"file_id": "uuid", "filename": "CV.pdf", "source_path": "/abs/path"}
    created_at: str = ""
    indexed: bool = False
    index_wait_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "applicant_name": self.applicant_name,
            "folder_name": self.folder_name,
            "kb_id": self.kb_id,
            "kb_name": self.kb_name,
            "file_ids": self.file_ids,
            "created_at": self.created_at,
            "indexed": self.indexed,
            "index_wait_seconds": self.index_wait_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RAGMetadata":
        return cls(
            applicant_name=data.get("applicant_name", ""),
            folder_name=data.get("folder_name", ""),
            kb_id=data.get("kb_id", ""),
            kb_name=data.get("kb_name", ""),
            file_ids=data.get("file_ids", []),
            created_at=data.get("created_at", ""),
            indexed=data.get("indexed", False),
            index_wait_seconds=data.get("index_wait_seconds", 0),
        )

    def save(self, results_path: Path):
        """Save metadata to the applicant's results directory."""
        results_path.mkdir(parents=True, exist_ok=True)
        filepath = results_path / METADATA_FILENAME
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"RAG metadata saved: {filepath}")

    @classmethod
    def load(cls, results_path: Path) -> Optional["RAGMetadata"]:
        """Load metadata if it exists. Returns None if not found."""
        filepath = results_path / METADATA_FILENAME
        if not filepath.exists():
            return None
        try:
            with open(filepath) as f:
                return cls.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load RAG metadata from {filepath}: {e}")
            return None


class ApplicantKBManager:
    """
    Manages per-applicant knowledge bases on GenAI Studio.

    Each applicant gets their own KB containing all their application
    documents. This enables RAG-grounded evaluation where the LLM can
    access the full text of any document without truncation.

    The manager is idempotent: if a KB already exists for an applicant
    (based on saved metadata), it returns the existing KB ID rather
    than creating a duplicate.
    """

    def __init__(
        self,
        client: LLMClient,
        index_wait: int = DEFAULT_INDEX_WAIT,
        upload_settle: int = DEFAULT_UPLOAD_SETTLE,
    ):
        """
        Args:
            client: LLMClient with RAG methods available.
            index_wait: Seconds to wait after linking for indexing.
            upload_settle: Seconds to wait after upload before linking.
        """
        self.client = client
        self.index_wait = index_wait
        self.upload_settle = upload_settle

    # ════════════════════════════════════════════════════════════════════
    # CREATE / GET
    # ════════════════════════════════════════════════════════════════════

    def get_or_create_kb(
        self,
        applicant_name: str,
        folder_name: str,
        source_folder: Path,
        results_path: Path,
        force_recreate: bool = False,
    ) -> str:
        """
        Get existing KB or create a new one for an applicant.

        This is the primary entry point. It:
        1. Checks for existing metadata (idempotent)
        2. Validates the existing KB still exists on the server
        3. If no valid KB, creates one from scratch
        4. Saves metadata for future reuse

        Args:
            applicant_name: Display name (e.g., "Alice Smith").
            folder_name: Directory name (e.g., "Alice_Smith").
            source_folder: Path to the applicant's document folder.
            results_path: Path to save metadata (e.g., data/results/Alice_Smith/).
            force_recreate: If True, delete existing KB and start fresh.

        Returns:
            KB ID (UUID string) ready for use in collections=[kb_id].
        """
        # Check for existing KB
        if not force_recreate:
            existing = self._get_existing_kb(results_path)
            if existing:
                logger.info(
                    f"Reusing existing KB for {applicant_name}: {existing}"
                )
                return existing

        # If force_recreate, clean up old KB first
        if force_recreate:
            self._cleanup_existing(results_path)

        # Create new KB
        return self._create_fresh_kb(
            applicant_name, folder_name, source_folder, results_path
        )

    def get_kb_id(
        self, folder_name: str, results_path: Path
    ) -> Optional[str]:
        """
        Get the KB ID for an applicant if it exists.

        Lightweight check — just reads metadata, no server calls.

        Args:
            folder_name: The applicant's folder name.
            results_path: Path to the applicant's results directory.

        Returns:
            KB ID string, or None if no KB exists.
        """
        meta = RAGMetadata.load(results_path)
        if meta and meta.kb_id:
            return meta.kb_id
        return None

    def get_all_kb_ids(self, results_dir: Path) -> dict[str, str]:
        """
        Scan all applicant results and return {folder_name: kb_id} mapping.

        Useful for the chat interface to know which applicants have KBs.

        Args:
            results_dir: Top-level results directory containing per-applicant folders.

        Returns:
            Dict mapping folder_name -> kb_id for all applicants with KBs.
        """
        mapping = {}
        if not results_dir.exists():
            return mapping

        for folder in sorted(results_dir.iterdir()):
            if not folder.is_dir():
                continue
            meta = RAGMetadata.load(folder)
            if meta and meta.kb_id:
                mapping[meta.folder_name or folder.name] = meta.kb_id

        return mapping

    # ════════════════════════════════════════════════════════════════════
    # CREATION INTERNALS
    # ════════════════════════════════════════════════════════════════════

    def _get_existing_kb(self, results_path: Path) -> Optional[str]:
        """
        Check if a valid KB already exists for this applicant.

        Validates the KB still exists on the server (it may have been
        deleted externally).
        """
        meta = RAGMetadata.load(results_path)
        if not meta or not meta.kb_id:
            return None

        # Verify KB still exists on server
        try:
            self.client.get_knowledge_base(meta.kb_id)
            return meta.kb_id
        except Exception as e:
            logger.warning(
                f"Saved KB {meta.kb_id} no longer exists on server: {e}. "
                f"Will recreate."
            )
            return None

    def _create_fresh_kb(
        self,
        applicant_name: str,
        folder_name: str,
        source_folder: Path,
        results_path: Path,
    ) -> str:
        """
        Create a brand new KB: upload files, create KB, link, wait.

        Returns the KB ID.
        """
        from datetime import datetime

        kb_name = f"Applicant: {applicant_name}"
        description = (
            f"Application documents for {applicant_name} "
            f"(VAP Search - Statistics)"
        )

        # Step 1: Discover uploadable files
        files_to_upload = self._discover_files(source_folder)
        if not files_to_upload:
            raise RuntimeError(
                f"No uploadable files found in {source_folder}. "
                f"Supported: {', '.join(sorted(UPLOADABLE_EXTENSIONS))}"
            )

        logger.info(
            f"Found {len(files_to_upload)} files to upload for {applicant_name}"
        )

        # Step 2: Upload files
        uploaded = []
        for filepath in files_to_upload:
            try:
                info = self.client.upload_file(str(filepath))
                uploaded.append({
                    "file_id": info.id,
                    "filename": info.filename,
                    "source_path": str(filepath),
                })
                logger.info(f"  ✅ Uploaded: {filepath.name} -> {info.id}")
            except Exception as e:
                logger.error(f"  ❌ Upload failed for {filepath.name}: {e}")

        if not uploaded:
            raise RuntimeError(
                f"All file uploads failed for {applicant_name}"
            )

        # Brief pause to let uploads settle on server
        if self.upload_settle > 0:
            logger.info(
                f"  Waiting {self.upload_settle}s for uploads to settle..."
            )
            time.sleep(self.upload_settle)

        # Step 3: Create knowledge base
        kb = self.client.create_knowledge_base(kb_name, description)
        logger.info(f"  Created KB: {kb.name} -> {kb.id}")

        # Step 4: Link files to KB
        linked_count = 0
        for file_entry in uploaded:
            try:
                self.client.add_file_to_knowledge_base(
                    kb.id, file_entry["file_id"]
                )
                linked_count += 1
                logger.info(
                    f"  🔗 Linked: {file_entry['filename']} -> KB"
                )
            except Exception as e:
                logger.error(
                    f"  ❌ Link failed for {file_entry['filename']}: {e}"
                )

        if linked_count == 0:
            # Clean up the empty KB
            try:
                self.client.delete_knowledge_base(kb.id)
            except Exception:
                pass
            raise RuntimeError(
                f"Could not link any files to KB for {applicant_name}"
            )

        # Step 5: Wait for indexing
        if self.index_wait > 0:
            logger.info(
                f"  ⏳ Waiting {self.index_wait}s for indexing..."
            )
            time.sleep(self.index_wait)

        # Step 6: Save metadata
        meta = RAGMetadata(
            applicant_name=applicant_name,
            folder_name=folder_name,
            kb_id=kb.id,
            kb_name=kb_name,
            file_ids=uploaded,
            created_at=datetime.now().isoformat(),
            indexed=True,
            index_wait_seconds=self.index_wait,
        )
        meta.save(results_path)

        logger.info(
            f"  ✅ KB ready for {applicant_name}: "
            f"{linked_count}/{len(uploaded)} files indexed"
        )

        return kb.id

    def _discover_files(self, source_folder: Path) -> list[Path]:
        """
        Find all uploadable files in an applicant's folder.

        Returns sorted list of file paths with supported extensions.
        Does not recurse into subdirectories.
        """
        if not source_folder.is_dir():
            return []

        files = []
        for f in sorted(source_folder.iterdir()):
            if f.is_file() and f.suffix.lower() in UPLOADABLE_EXTENSIONS:
                # Skip hidden files and very small files (likely empty)
                if not f.name.startswith(".") and f.stat().st_size > 100:
                    files.append(f)

        return files

    # ════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ════════════════════════════════════════════════════════════════════

    def cleanup_applicant_kb(
        self,
        results_path: Path,
        delete_files: bool = True,
    ) -> bool:
        """
        Delete an applicant's KB and optionally their uploaded files.

        Also removes the local metadata file.

        Args:
            results_path: Path to the applicant's results directory.
            delete_files: If True, also delete uploaded files from server.

        Returns:
            True if cleanup succeeded (or nothing to clean).
        """
        meta = RAGMetadata.load(results_path)
        if not meta:
            logger.info("No RAG metadata found — nothing to clean up")
            return True

        success = True

        # Delete KB
        if meta.kb_id:
            try:
                self.client.delete_knowledge_base(meta.kb_id)
                logger.info(f"  ✅ Deleted KB: {meta.kb_id}")
            except Exception as e:
                logger.warning(f"  ⚠️ KB deletion failed: {e}")
                success = False

        # Delete uploaded files
        if delete_files:
            for file_entry in meta.file_ids:
                file_id = file_entry.get("file_id", "")
                if file_id:
                    try:
                        self.client.delete_file(file_id)
                        logger.info(
                            f"  ✅ Deleted file: {file_entry.get('filename', file_id)}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"  ⚠️ File deletion failed for {file_id}: {e}"
                        )
                        success = False

        # Remove local metadata
        meta_file = results_path / METADATA_FILENAME
        if meta_file.exists():
            meta_file.unlink()
            logger.info(f"  ✅ Removed metadata: {meta_file}")

        return success

    def cleanup_all(
        self,
        results_dir: Path,
        delete_files: bool = True,
    ) -> dict[str, bool]:
        """
        Clean up all applicant KBs in a results directory.

        Args:
            results_dir: Top-level results directory.
            delete_files: If True, also delete uploaded files.

        Returns:
            Dict mapping folder_name -> cleanup_success.
        """
        results = {}

        for folder in sorted(results_dir.iterdir()):
            if not folder.is_dir():
                continue

            meta = RAGMetadata.load(folder)
            if meta:
                logger.info(f"Cleaning up KB for: {meta.applicant_name}")
                success = self.cleanup_applicant_kb(
                    folder, delete_files=delete_files
                )
                results[folder.name] = success

        return results

    def _cleanup_existing(self, results_path: Path):
        """Clean up existing KB before recreation. Logs but doesn't raise."""
        try:
            self.cleanup_applicant_kb(results_path, delete_files=True)
        except Exception as e:
            logger.warning(f"Cleanup of existing KB failed: {e}")

    # ════════════════════════════════════════════════════════════════════
    # STATUS / DIAGNOSTICS
    # ════════════════════════════════════════════════════════════════════

    def status(self, results_path: Path) -> dict:
        """
        Get the RAG status for a single applicant.

        Returns:
            Dict with keys: has_kb, kb_id, kb_exists_on_server,
            file_count, applicant_name.
        """
        meta = RAGMetadata.load(results_path)
        if not meta:
            return {
                "has_kb": False,
                "kb_id": None,
                "kb_exists_on_server": False,
                "file_count": 0,
                "applicant_name": None,
            }

        # Check server
        server_ok = False
        if meta.kb_id:
            try:
                self.client.get_knowledge_base(meta.kb_id)
                server_ok = True
            except Exception:
                pass

        return {
            "has_kb": True,
            "kb_id": meta.kb_id,
            "kb_exists_on_server": server_ok,
            "file_count": len(meta.file_ids),
            "applicant_name": meta.applicant_name,
        }

    def status_all(self, results_dir: Path) -> list[dict]:
        """Get RAG status for all applicants."""
        statuses = []
        for folder in sorted(results_dir.iterdir()):
            if folder.is_dir():
                s = self.status(folder)
                s["folder_name"] = folder.name
                statuses.append(s)
        return statuses