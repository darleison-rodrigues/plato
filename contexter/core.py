"""
Main Pipeline: Orchestrates PDF → MD → Vector Store → Knowledge Graph
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from .parser import PDFProcessor
from .llm import OllamaClient
from .store import VectorStore
from .graph import KnowledgeGraphBuilder
from .config import get_config


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.config = get_config()
        self.pdf_processor = PDFProcessor()
        self.llm_client = OllamaClient()
        self.vector_store = VectorStore()
        self.kg_builder = KnowledgeGraphBuilder()
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_pdf(
        self,
        pdf_path: str,
        store_in_vector_db: bool = True,
        build_kg: bool = True
    ) -> Dict:
        """
        Complete pipeline for a single PDF
        
        Args:
            pdf_path: Path to PDF file
            store_in_vector_db: Whether to store in Chroma
            build_kg: Whether to build knowledge graph
        
        Returns:
            Dictionary with all outputs and statistics
        """
        pdf_name = Path(pdf_path).stem
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Convert PDF to Markdown
        print("Step 1: Converting PDF to Markdown")
        markdown_text, md_path = self.pdf_processor.convert_to_markdown(pdf_path)
        
        # Step 2: Store in vector database
        if store_in_vector_db:
            print("\nStep 2: Storing in vector database")
            metadata = self.pdf_processor.get_pdf_metadata(pdf_path)
            self.vector_store.add_document(pdf_name, markdown_text, metadata)
        
        # Step 3: Extract entities and relations
        print("\nStep 3: Extracting entities and relations using LLM")
        entities = self.llm_client.extract_entities(markdown_text)
        relations = self.llm_client.extract_relations(markdown_text)
        
        # Save extracted data
        extraction_data = {
            "entities": entities,
            "relations": relations
        }
        
        entities_path = self.output_dir / f"{pdf_name}_entities.json"
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2)
        
        print(f"✓ Extracted {sum(len(v) for v in entities.values())} entities")
        print(f"✓ Extracted {len(relations)} relations")
        print(f"✓ Saved to {entities_path}")
        
        # Step 4: Build knowledge graph
        graph_path = ""
        viz_path = ""
        kg_stats = {}
        
        if build_kg:
            print("\nStep 4: Building knowledge graph")
            G = self.kg_builder.build_graph(entities, relations)
            graph_path = self.kg_builder.save_graph(G, pdf_name)
            viz_path = self.kg_builder.visualize(G, pdf_name)
            kg_stats = self.kg_builder.get_statistics(G)
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Markdown:       {md_path}")
        print(f"Entities:       {entities_path}")
        if graph_path:
            print(f"Graph:          {graph_path}")
        if viz_path:
            print(f"Visualization:  {viz_path}")
        print(f"\nEntity counts:")
        for entity_type, entity_list in entities.items():
            print(f"  {entity_type}: {len(entity_list)}")
        if kg_stats:
            print(f"\nGraph stats:")
            print(f"  Nodes: {kg_stats['num_nodes']}")
            print(f"  Edges: {kg_stats['num_edges']}")
            print(f"  Density: {kg_stats['density']:.4f}")
        
        return {
            "pdf_name": pdf_name,
            "markdown_path": md_path,
            "entities_path": str(entities_path),
            "graph_path": graph_path,
            "visualization_path": viz_path,
            "entities": entities,
            "relations": relations,
            "kg_stats": kg_stats
        }
    
    def process_directory(
        self,
        directory: str,
        store_in_vector_db: bool = True,
        build_kg: bool = True
    ) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Path to directory containing PDFs
            store_in_vector_db: Whether to store in Chroma
            build_kg: Whether to build knowledge graphs
        
        Returns:
            List of result dictionaries
        """
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files\n")
        results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]")
            try:
                result = self.process_pdf(
                    str(pdf_file),
                    store_in_vector_db=store_in_vector_db,
                    build_kg=build_kg
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error processing {pdf_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Overall summary
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successfully processed: {len(results)}/{len(pdf_files)} files")
        
        if store_in_vector_db:
            stats = self.vector_store.get_stats()
            print(f"\nVector store stats:")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Total chunks: {stats['total_chunks']}")
        
        return results
    
    def _search_knowledge_graph(self, query: str) -> List[Dict]:
        """
        Search for entities and relations in the knowledge graph
        """
        # Step 1: Extract entities from query to understand what to look for
        print(f"  Analyzing query for entities...")
        query_entities = self.llm_client.extract_entities(query)
        
        entities_of_interest = []
        for entity_list in query_entities.values():
            entities_of_interest.extend(entity_list)
            
        if not entities_of_interest:
            return []
            
        print(f"  Looking for relationships involving: {', '.join(entities_of_interest[:3])}...")
        
        # Step 2: Load all available graphs and search for entities
        kg_results = []
        graph_files = list(self.output_dir.glob("*_graph.json"))
        
        for graph_file in graph_files:
            try:
                G = self.kg_builder.load_graph(str(graph_file))
                doc_id = graph_file.stem.replace("_graph", "")
                
                for entity in entities_of_interest:
                    # Case insensitive check for entity in graph
                    matches = [node for node in G.nodes() if entity.lower() in node.lower()]
                    
                    for match in matches:
                        # Get neighbors and relations
                        for neighbor in G.neighbors(match):
                            edge_data = G.get_edge_data(match, neighbor)
                            kg_results.append({
                                "source": doc_id,
                                "subject": match,
                                "relation": edge_data.get('relation', 'connected to'),
                                "object": neighbor
                            })
                        # Also check incoming edges
                        for predecessor in G.predecessors(match):
                            edge_data = G.get_edge_data(predecessor, match)
                            kg_results.append({
                                "source": doc_id,
                                "subject": predecessor,
                                "relation": edge_data.get('relation', 'connected to'),
                                "object": match
                            })
            except Exception as e:
                print(f"  Warning: Error searching graph {graph_file.name}: {e}")
                
        return kg_results

    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """
        Hybrid Search: Combines Knowledge Graph relations with Vector search
        """
        print(f"\n--- Hybrid Search: '{query}' ---")
        
        # 1. Knowledge Graph Search
        print("1. Querying Knowledge Graph...")
        kg_results = self._search_knowledge_graph(query)
        
        if kg_results:
            print(f"   Found {len(kg_results)} relevant relationships in KG:")
            # Unique relationships
            seen = set()
            for rel in kg_results[:5]: # Show top 5
                rel_str = f"   - [{rel['source']}] {rel['subject']} --({rel['relation']})--> {rel['object']}"
                if rel_str not in seen:
                    print(rel_str)
                    seen.add(rel_str)
        else:
            print("   No direct relationships found in KG.")
            
        # 2. Vector Semantic Search
        print("\n2. Querying Vector Store (Semantic Search)...")
        results = self.vector_store.search(query, n_results)
        
        print(f"   Found {len(results['documents'][0])} relevant text chunks:")
        for i, (doc, metadata, distance) in enumerate(
            zip(results['documents'][0], results['metadatas'][0], results['distances'][0]),
            1
        ):
            print(f"\n   {i}. Document: {metadata['doc_id']} (chunk {metadata['chunk_index']})")
            print(f"      Distance: {distance:.4f}")
            print(f"      Preview: {doc[:150]}...")
            
        # Combine results
        return {
            "query": query,
            "vector_results": results,
            "graph_results": kg_results
        }
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return self.vector_store.get_stats()
    
    def list_documents(self) -> List[str]:
        """List all documents in the vector store"""
        return self.vector_store.list_documents()
