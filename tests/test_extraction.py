"""
Tests for document extraction module.

Run with: pytest tests/test_extraction.py
"""

import pytest
from pathlib import Path


class TestPDFExtraction:
    """Tests for PDF text extraction."""
    
    def test_placeholder(self):
        """Placeholder test - implement with actual test PDFs."""
        # TODO: Add test PDFs to tests/fixtures/
        # TODO: Test extraction quality
        assert True


class TestDOCXExtraction:
    """Tests for DOCX text extraction."""
    
    def test_placeholder(self):
        """Placeholder test - implement with actual test DOCXs."""
        assert True


class TestTextCleaning:
    """Tests for text cleaning utilities."""
    
    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        from src.extraction.text_cleaner import clean_text
        
        input_text = "Hello    world\n\n\n\nNew paragraph"
        expected = "Hello world\n\nNew paragraph"
        
        assert clean_text(input_text) == expected
