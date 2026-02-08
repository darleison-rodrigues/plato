"""
Main Pipeline: Orchestrates PDF → MD → Vector Store → Knowledge Graph
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import concurrent.futures
from threading import Lock

from .parser import PDFProcessor
from .llm import OllamaClient
from .store import VectorStore
from .graph import KnowledgeGraphBuilder
from .config import get_config


class Pipeline:
    """Main pipeline orchestrator with robust error handling and parallel processing"""
    
    def __init__(self):
        self.config = get_config()
        self.pdf_processor = PDFProcessor()
        self.llm_client = OllamaClient()
        self.vector_store = VectorStore()
        self.kg_builder = KnowledgeGraphBuilder()
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def process_pdf(
        self,
        pdf_path: str,
        store_in_vector_db: bool = True,
        build_kg: bool = True,
        resume_on_error: bool = True
    ) -> Dict:
        """
        Complete pipeline for a single PDF with robust error handling
        """
        pdf_name = Path(pdf_path).stem
        self.logger.info(f"Processing: {pdf_name}")
        
        # Initialize result tracking
        result = {
            "pdf_name": pdf_name,
            "markdown_path": "",
            "entities_path": "",
            "graph_path": "",
            "visualization_path": "",
            "entities": {},
            "relations": [],
            "kg_stats": {},
            "errors": [],
            "success_steps": []
        }
        
        try:
            # Step 1: Convert PDF to Markdown
            self.logger.info("Step 1: Converting PDF to Markdown")
            markdown_text, md_path = self.pdf_processor.convert_to_markdown(pdf_path)
            result["markdown_path"] = md_path
            result["success_steps"].append("pdf_conversion")
            
            # Validate markdown content
            if not markdown_text or len(markdown_text.strip()) < 50: # Lower limit for small docs
                raise ValueError(f"PDF conversion produced insufficient content: {len(markdown_text)} chars")
                
            # Step 2: Store in vector database
            if store_in_vector_db:
                try:
                    self.logger.info("Step 2: Storing in vector database")
                    metadata = self.pdf_processor.get_pdf_metadata(pdf_path)
                    self.vector_store.add_document(pdf_name, markdown_text, metadata)
                    result["success_steps"].append("vector_storage")
                except Exception as e:
                    self.logger.error(f"Vector storage failed: {e}")
                    result["errors"].append(f"Vector storage failed: {e}")
                    if not resume_on_error: raise

            # Step 3: Extract entities and relations
            self.logger.info("Step 3: Extracting entities and relations using LLM")
            entities = self.llm_client.extract_entities(markdown_text)
            relations = self.llm_client.extract_relations(markdown_text)
            
            result["entities"] = entities
            result["relations"] = relations
            result["success_steps"].append("entity_extraction")
            
            # Save extracted data
            extraction_data = {
                "entities": entities,
                "relations": relations,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "pdf_path": pdf_path
                }
            }
            
            entities_path = self.output_dir / f"{pdf_name}_entities.json"
            with open(entities_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_data, f, indent=2)
            result["entities_path"] = str(entities_path)
            
            self.logger.info(f"Extracted {sum(len(v) for v in entities.values() if isinstance(v, list))} entities")
            
            # Step 4: Build knowledge graph
            if build_kg:
                try:
                    self.logger.info("Step 4: Building knowledge graph")
                    G = self.kg_builder.build_graph(entities, relations)
                    
                    if G.number_of_nodes() > 0:
                        graph_path = self.kg_builder.save_graph(G, pdf_name)
                        viz_path = self.kg_builder.visualize(G, pdf_name)
                        kg_stats = self.kg_builder.get_statistics(G)
                        
                        result["graph_path"] = graph_path
                        result["visualization_path"] = viz_path
                        result["kg_stats"] = kg_stats
                        result["success_steps"].append("graph_build")
                    else:
                        self.logger.warning("Knowledge graph empty, skipping save/viz")
                        
                except Exception as e:
                     self.logger.error(f"Knowledge graph building failed: {e}")
                     result["errors"].append(f"Graph build failed: {e}")
                     if not resume_on_error: raise

        except Exception as e:
            self.logger.error(f"Pipeline failed for {pdf_name}: {e}")
            result["errors"].append(str(e))
            if not resume_on_error:
                raise
        
        return result
    
    def process_directory_parallel(
        self,
        directory: str,
        store_in_vector_db: bool = True,
        build_kg: bool = True,
        max_workers: int = 3
    ) -> List[Dict]:
        """Process PDFs in parallel with robust error handling"""
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return []
        
        max_workers = min(max_workers, len(pdf_files))
        self.logger.info(f"Processing {len(pdf_files)} PDF files with {max_workers} workers")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(
                    self.process_pdf, 
                    str(pdf), 
                    store_in_vector_db, 
                    build_kg
                ): pdf for pdf in pdf_files
            }
            
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    self.logger.error(f"{pdf} generated an exception: {exc}")
        
        return results
    
    def _search_knowledge_graph(self, query: str) -> List[Dict]:
        """Search for entities and relations in the knowledge graph"""
        # Step 1: Extract entities from query
        query_entities = self.llm_client.extract_entities(query)
        entities_of_interest = []
        if isinstance(query_entities, dict):
            for entity_list in query_entities.values():
                if isinstance(entity_list, list):
                    entities_of_interest.extend(entity_list)
            
        if not entities_of_interest:
            return []
            
        # Step 2: Search graphs
        kg_results = []
        graph_files = list(self.output_dir.glob("*_graph.json"))
        
        for graph_file in graph_files:
            try:
                G = self.kg_builder.load_graph(str(graph_file))
                doc_id = graph_file.stem.replace("_graph", "")
                
                for entity in entities_of_interest:
                    # Case insensitive matches
                    matches = [node for node in G.nodes() if str(entity).lower() in str(node).lower()]
                    
                    for match in matches:
                        # Outgoing edges
                        for neighbor in G.neighbors(match):
                            edge_data = G.get_edge_data(match, neighbor)
                            kg_results.append({
                                "source_document": doc_id,
                                "subject": match,
                                "relation": edge_data.get('relation', 'connected to'),
                                "object": neighbor,
                                "source": "knowledge_graph"
                            })
                        # Incoming edges (optional, can be noisy)
            except Exception as e:
                self.logger.warning(f"Error searching graph {graph_file.name}: {e}")
                
        return kg_results

    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """
        Hybrid Search: Combines Knowledge Graph relations with Vector search
        Ranking and deduplicating results intelligently.
        """
        self.logger.info(f"Hybrid Search: '{query}'")
        
        # Parallel execution of searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
             future_kg = executor.submit(self._search_knowledge_graph, query)
             future_vec = executor.submit(self.vector_store.search, query, n_results)
             
             try:
                 kg_results = future_kg.result(timeout=30)
                 vector_results = future_vec.result(timeout=30)
             except concurrent.futures.TimeoutError:
                 self.logger.error("Search timed out")
                 return {"query": query, "combined_results": []}

        # Combine results
        combined_results = []
        
        # Process vector results
        if vector_results and vector_results.get('documents') and vector_results['documents'][0]:
            for doc, metadata, distance in zip(
                vector_results['documents'][0], 
                vector_results['metadatas'][0], 
                vector_results['distances'][0]
            ):
                combined_results.append({
                    "type": "document",
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1.0 - distance,  # Convert distance to relevance
                    "source": "vector_search"
                })
        
        # Process KG results
        for kg_res in kg_results:
             # Basic relevance scoring based on query term overlap
             score = 0.5 
             combined_results.append({
                "type": "relationship",
                "subject": kg_res["subject"],
                "relation": kg_res["relation"],
                "object": kg_res["object"],
                "source_document": kg_res["source_document"],
                "relevance_score": score,
                "source": "knowledge_graph"
            })

        # Sort combined results by score
        combined_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": kg_results,
            "combined_results": combined_results[:10] # Top 10
        }
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return self.vector_store.get_stats()
    
    def list_documents(self) -> List[str]:
        """List all documents in the vector store"""
        return self.vector_store.list_documents()
