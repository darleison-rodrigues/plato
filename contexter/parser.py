import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
from docling.document_converter import DocumentConverter
from .config import get_config


class PDFProcessor:
    """Handles PDF conversion to Markdown and metadata extraction"""
    
    MARKDOWN_EXTENSION = ".md"
    
    def __init__(self):
        self.config = get_config()
        self.converter = DocumentConverter()
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_to_markdown(self, pdf_path: str) -> Tuple[str, str]:
        """
        Convert PDF to Markdown using Docling with validation.
        
        Args:
            pdf_path: Absolute path to the PDF file.
            
        Returns:
            Tuple containing (markdown_content, output_file_path)
            
        Raises:
            FileNotFoundError: If PDF does not exist.
            ValueError: If conversion results in empty content.
            RuntimeError: If conversion fails.
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        print(f"Converting {pdf_path_obj.name} to Markdown...")
        
        try:
            # Convert document
            result = self.converter.convert(str(pdf_path_obj))
            markdown_text = result.document.export_to_markdown()
            
            if not markdown_text or not markdown_text.strip():
                raise ValueError(f"PDF conversion produced empty content: {pdf_path}")
            
            # Save output
            md_filename = f"{pdf_path_obj.stem}{self.MARKDOWN_EXTENSION}"
            md_path = self.output_dir / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
                
            return markdown_text, str(md_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF {pdf_path}: {str(e)}") from e
        
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract file system metadata from the PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary containing file metadata.
            
        Raises:
            FileNotFoundError: If file cannot be accessed.
        """
        path_obj = Path(pdf_path)
        try:
            stat = path_obj.stat()
            return {
                "source": str(path_obj.absolute()),
                "filename": path_obj.name,
                "doc_id": path_obj.stem,
                "modified_time": stat.st_mtime,
                "file_size": stat.st_size,
                "created_time": stat.st_ctime
            }
        except OSError as e:
            raise FileNotFoundError(f"Cannot access file metadata for {pdf_path}: {e}") from e

    def batch_convert(self, pdf_paths: List[str]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Convert multiple PDFs in batch and return (markdown, path, metadata) tuples.
        Continuing on individual failures (fault tolerance).
        
        Args:
            pdf_paths: List of paths to PDF files.
            
        Returns:
            List of tuples containing (markdown_content, output_path, metadata).
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                markdown, path = self.convert_to_markdown(pdf_path)
                metadata = self.get_pdf_metadata(pdf_path)
                results.append((markdown, path, metadata))
            except Exception as e:
                # Log error and continue with batch processing so one bad file doesn't stop the pipeline
                print(f"Failed to process {pdf_path}: {e}")
                continue
        return results
