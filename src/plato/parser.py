from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import logging
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF loading using LlamaIndex native readers for best performance
    and minimal dependencies.
    """

    def __init__(self):
        pass

    def convert_to_markdown(self, pdf_path: str) -> Tuple[str, str]:
        """
        Load PDF text using LlamaIndex.
        Returns: (text_content, output_path)
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            logger.info(f"Loading {pdf_path} using LlamaIndex...")
            # Use SimpleDirectoryReader for robust local loading
            reader = SimpleDirectoryReader(input_files=[str(pdf_path_obj)])
            documents = reader.load_data()
            
            # Combine text from pages
            full_text = "\n\n".join([doc.text for doc in documents])
            
            # Save locally as reference (optional, but keeps interface consistent)
            output_path = pdf_path_obj.with_suffix('.md')
            output_path.write_text(full_text, encoding='utf-8')
            
            return full_text, str(output_path)
            
        except Exception as e:
            logger.error(f"LlamaIndex loading failed for {pdf_path}: {e}")
            raise

    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract basic metadata"""
        return {
            "source": pdf_path,
            "processor": "llama_index"
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
