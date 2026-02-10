import fitz  # PyMuPDF
from pathlib import Path
from typing import Tuple, List, Optional
from plato.core.errors import PDFProcessingError

class PDFProcessor:
    """
    Handles efficient and safe PDF text extraction and analysis using PyMuPDF.
    """
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path).expanduser()
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

    def extract_text(self) -> str:
        """
        Extracts all text from the PDF file efficiently.
        
        Raises:
            PDFProcessingError: If text extraction fails.
        """
        page_texts = []
        try:
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    page_texts.append(page.get_text())
        except Exception as e:
            raise PDFProcessingError(f"Failed to process {self.pdf_path}: {e}")

        return "".join(page_texts)

    def analyze_content(self) -> Tuple[str, bool]:
        """
        Extracts all text and determines if the PDF is likely scanned (no text) in a single pass.
        
        Returns:
            A tuple containing:
            - The full extracted text (str).
            - A flag indicating if the document is scanned (bool).
            
        Raises:
            PDFProcessingError: If processing fails.
        """
        page_texts = []
        is_text_found = False
        try:
            with fitz.open(self.pdf_path) as doc:
                if not doc.page_count:
                    # An empty PDF is effectively scanned or invalid
                    return "", True
                    
                for page in doc:
                    text = page.get_text()
                    if not is_text_found and text.strip():
                        is_text_found = True
                    page_texts.append(text)
        except Exception as e:
            raise PDFProcessingError(f"Failed to analyze {self.pdf_path}: {e}")

        full_text = "".join(page_texts)
        is_scanned = not is_text_found

        return full_text, is_scanned

    @staticmethod
    def is_scanned_check(pdf_path: str | Path) -> bool:
        """
        Quick check if a PDF is scanned without full extraction.
        Useful for UI validation before heavy processing.
        """
        path = Path(pdf_path).expanduser()
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    if page.get_text().strip():
                        return False
            return True
        except Exception as e:
            raise PDFProcessingError(f"Failed to check if PDF is scanned {path}: {e}")

