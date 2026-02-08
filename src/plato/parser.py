from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF to Markdown conversion using Docling.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def convert_to_markdown(self, pdf_path: str) -> Tuple[str, str]:
        """
        Convert a PDF file to Markdown using Docling.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Tuple[str, str]: (markdown_content, output_path)
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            logger.info(f"Converting {pdf_path} using Docling...")
            result = self.converter.convert(pdf_path_obj)
            markdown_text = result.document.export_to_markdown()
            
            # Save markdown locally for reference
            output_path = pdf_path_obj.with_suffix('.md')
            output_path.write_text(markdown_text, encoding='utf-8')
            
            return markdown_text, str(output_path)
            
        except Exception as e:
            logger.error(f"Docling conversion failed for {pdf_path}: {e}")
            raise

    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract basic metadata from PDF"""
        # Docling handles this internally, but we can add a lightweight 
        # metadata extractor here if needed.
        return {
            "source": pdf_path,
            "processor": "docling"
        }

class DocumentParser:
    """
    Legacy wrapper for directory loading if needed, 
    but mainly we use PDFProcessor for individual files.
    """
    def __init__(self):
        self.processor = PDFProcessor()

    def load_data(self, input_dir: Path) -> List[Any]:
        """
        Load all supported documents from a directory.
        Returns a list of LlamaIndex-compatible Documents.
        """
        from llama_index.core import Document
        
        docs = []
        input_path = Path(input_dir)
        
        # Process PDFs
        for pdf_file in input_path.glob("*.pdf"):
            try:
                text, _ = self.processor.convert_to_markdown(str(pdf_file))
                docs.append(Document(text=text, metadata={"source": str(pdf_file)}))
            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")
                
        # Process MDs
        for md_file in input_path.glob("*.md"):
            try:
                text = md_file.read_text(encoding='utf-8')
                docs.append(Document(text=text, metadata={"source": str(md_file)}))
            except Exception as e:
                logger.error(f"Failed to load {md_file}: {e}")
                
        return docs
