from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document

class DocumentParser:
    """
    Handles loading and parsing of documents from a directory using LlamaIndex.
    """

    def __init__(self):
        pass

    def load_data(self, input_dir: Path) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            input_dir: The directory to read documents from.
            
        Returns:
            A list of LlamaIndex Document objects.
        """
        if not input_dir.is_dir():
            raise ValueError(f"Input path must be a directory. Got: {input_dir}")

        reader = SimpleDirectoryReader(input_dir=str(input_dir), recursive=True)
        documents = reader.load_data(show_progress=True)
        
        if not documents:
            print(f"Warning: No documents found in {input_dir}")
            
        return documents

