import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

def extract_text_from_pdf(filepath: Path) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    
    Args:
        filepath: Path to the PDF file.
        
    Returns:
        The extracted text content.
    """
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def is_scanned(filepath: Path) -> bool:
    """
    Detects if a PDF is likely scanned (contains mostly images/no text).
    """
    doc = fitz.open(filepath)
    for page in doc:
        if page.get_text().strip():
            return False
    return True
