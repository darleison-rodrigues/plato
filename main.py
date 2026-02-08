#!/usr/bin/env python3
"""
Command-line interface for the PDF-to-KG Pipeline
"""
import argparse
import sys
from pathlib import Path
from contexter.core import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="PDF to Knowledge Graph Pipeline with Ollama and Chroma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF
  python main.py process document.pdf
  
  # Process directory of PDFs
  python main.py process pdfs/ --batch
  
  # Skip knowledge graph generation (faster)
  python main.py process document.pdf --no-kg
  
  # Search documents
  python main.py search "machine learning"
  
  # List all documents in vector store
  python main.py list
  
  # Show vector store statistics
  python main.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process PDF(s)')
    process_parser.add_argument('input', help='PDF file or directory')
    process_parser.add_argument('--batch', action='store_true', help='Process directory of PDFs')
    process_parser.add_argument('--no-vector', action='store_true', help='Skip vector store')
    process_parser.add_argument('--no-kg', action='store_true', help='Skip knowledge graph')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-n', '--num-results', type=int, default=5, help='Number of results')
    
    # List command
    subparsers.add_parser('list', help='List all documents in vector store')
    
    # Stats command
    subparsers.add_parser('stats', help='Show vector store statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = Pipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nMake sure:")
        print("1. config.yaml exists")
        print("2. Ollama is running (ollama serve)")
        print("3. Required model is available (ollama pull llama3.2)")
        sys.exit(1)
    
    # Execute command
    if args.command == 'process':
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"Error: {args.input} does not exist")
            sys.exit(1)
        
        if args.batch or input_path.is_dir():
            # Process directory
            if not input_path.is_dir():
                print(f"Error: {args.input} is not a directory")
                sys.exit(1)
            
            pipeline.process_directory(
                str(input_path),
                store_in_vector_db=not args.no_vector,
                build_kg=not args.no_kg
            )
        else:
            # Process single file
            if not input_path.is_file() or input_path.suffix.lower() != '.pdf':
                print(f"Error: {args.input} is not a PDF file")
                sys.exit(1)
            
            pipeline.process_pdf(
                str(input_path),
                store_in_vector_db=not args.no_vector,
                build_kg=not args.no_kg
            )
    
    elif args.command == 'search':
        pipeline.search_documents(args.query, args.num_results)
    
    elif args.command == 'list':
        docs = pipeline.list_documents()
        if docs:
            print(f"\nDocuments in vector store ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                print(f"{i:3d}. {doc}")
        else:
            print("No documents in vector store")
    
    elif args.command == 'stats':
        stats = pipeline.get_vector_store_stats()
        print("\nVector Store Statistics:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total chunks: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
