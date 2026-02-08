import logging
from pathlib import Path
from typing import Dict, Any
from .base import BaseAgent
from plato.parser import PDFProcessor

logger = logging.getLogger(__name__)

class ScannerAgent(BaseAgent):
    """
    The 'Scanner': A lightweight agent that quickly peeks at documents
    to classify them and extract basic metadata.
    Uses the 1B model by default for speed.
    """
    
    def __init__(self):
        # Force 1B model for speed, regardless of global config
        # Fallback to standard model if 1B not found (handled by Ollama)
        super().__init__(model_override="llama3.2:1b") 
        self.processor = PDFProcessor()

    def run(self, file_path: Path) -> Dict[str, Any]:
        """
        Scans a PDF and returns metadata.
        """
        logger.info(f"Scanning {file_path.name}...")
        
        # 1. Extract raw text (limited to first 2000 chars for speed)
        # We use the existing PDFProcessor but minimal mode
        try:
            full_text = self.processor._extract_text_docling(file_path) # Access internal method or use public API
            # Heuristic: First 3000 chars usually contain Title, Abstract, Intro
            sample_text = full_text[:3000] 
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            return {"error": str(e), "file": file_path.name}

        # 2. Ask 1B model to classify
        prompt = f"""
        Analyze this document text and extracting the following JSON:
        {{
            "title": "Document Title",
            "authors": ["Author 1", "Author 2"],
            "type": "Paper" | "Report" | "Specification" | "Book" | "Other",
            "summary": "1-sentence summary of what this is",
            "keywords": ["tag1", "tag2"]
        }}

        TEXT:
        {sample_text}
        """

        metadata = self.generate_json(prompt, system="You are a librarian.")
        metadata['file_path'] = str(file_path)
        return metadata
