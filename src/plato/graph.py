import json
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
from .config import get_config


class KnowledgeGraphBuilder:
    """Builds and manages the knowledge graph based on extracted entities and relations"""
    
    def __init__(self):
        self.config = get_config()
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def build_graph(self, entities: Dict[str, List[str]], relations: List[Dict[str, str]]) -> nx.DiGraph:
        """Construct a NetworkX directed graph from extracted data with validation"""
        if not entities and not relations:
            raise ValueError("Both entities and relations cannot be empty")
        
        # Validate entities structure
        for entity_type, entity_list in entities.items():
            if not isinstance(entity_list, list):
                raise ValueError(f"Entity list for type '{entity_type}' must be a list")
            if not all(isinstance(e, str) for e in entity_list):
                raise ValueError(f"All entities for type '{entity_type}' must be strings")
        
        # Validate relations structure
        for rel in relations:
            if not isinstance(rel, dict):
                raise ValueError("Each relation must be a dictionary")
            required_fields = ['subject', 'object']
            missing = [field for field in required_fields if field not in rel or not rel[field]]
            if missing:
                raise ValueError(f"Relation missing required fields: {missing}")
            if 'relation' not in rel or not rel['relation']:
                rel['relation'] = "RELATED"  # Default relation type
        
        # Build graph with duplicate handling
        G = nx.DiGraph()
        
        # Add entities as nodes (with deduplication)
        for entity_type, entity_list in entities.items():
            unique_entities = list(set(entity_list))  # Remove duplicates
            for entity in unique_entities:
                G.add_node(entity, type=entity_type)
        
        # Add relations as edges
        seen_edges = set()  # Track duplicate edges
        for rel in relations:
            subject, relation, obj = rel['subject'], rel['relation'], rel['object']
            edge_key = (subject, obj)
            
            if edge_key not in seen_edges:
                if subject not in G:
                    G.add_node(subject, type="UNKNOWN")
                if obj not in G:
                    G.add_node(obj, type="UNKNOWN")
                    
                G.add_edge(subject, obj, relation=relation)
                seen_edges.add(edge_key)
                    
        return G
        
    def save_graph(self, G: nx.DiGraph, name: str) -> str:
        """Save the graph data to a JSON file"""
        graph_data = {
            "nodes": [{"id": n, "type": G.nodes[n].get("type", "UNKNOWN")} for n in G.nodes()],
            "edges": [{"source": u, "target": v, "relation": d.get("relation", "")} 
                     for u, v, d in G.edges(data=True)]
        }
        
        output_path = self.output_dir / f"{name}_graph.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        return str(output_path)
        
    def visualize(self, G: nx.DiGraph, name: str, max_nodes: int = 50) -> str:
        """Create a visualization of the graph with robust error handling"""
        try:
            if G.number_of_nodes() == 0:
                print(f"Warning: Graph '{name}' has no nodes to visualize")
                return ""
                
            # Validate max_nodes parameter
            if max_nodes <= 0:
                raise ValueError("max_nodes must be positive")
                
            # Create subgraph for large graphs
            if G.number_of_nodes() > max_nodes:
                node_degrees = dict(G.degree())
                if not node_degrees:
                    raise ValueError("Cannot compute node degrees")
                top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:max_nodes]
                G_sub = G.subgraph(top_nodes)
            else:
                G_sub = G
            
            plt.figure(figsize=(12, 8))
            try:
                pos = nx.spring_layout(G_sub, k=0.5, iterations=50)
                
                # Safe color mapping with fallback
                node_attrs = nx.get_node_attributes(G_sub, 'type')
                types = set(node_attrs.values()) if node_attrs else {"UNKNOWN"}
                
                color_map = plt.get_cmap('tab10')
                type_to_color = {}
                if types:
                    for i, t in enumerate(types):
                        type_to_color[t] = color_map(i / len(types))
                
                node_colors = [type_to_color.get(node_attrs.get(n, 'UNKNOWN'), 'lightgray') 
                              for n in G_sub.nodes()]
                
                nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=800, alpha=0.8)
                nx.draw_networkx_edges(G_sub, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5)
                nx.draw_networkx_labels(G_sub, pos, font_size=10, font_family='sans-serif')
                
                plt.title(f"Knowledge Graph: {name}")
                plt.axis('off')
                
                output_path = self.output_dir / f"{name}_graph.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                
                return str(output_path)
            finally:
                plt.close() # Ensure cleanup even if drawing fails
            
        except Exception as e:
            # Fallback cleanup just in case
            plt.close('all')
            raise RuntimeError(f"Failed to create visualization for graph '{name}': {e}") from e
        
    def get_statistics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate graph statistics"""
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
            "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False
        }

    def _assess_data_quality(self, G: nx.DiGraph) -> Dict[str, float]:
        """Assess data quality metrics for ML pipeline integration"""
        if G.number_of_nodes() == 0:
            return {"completeness": 0.0, "connectivity": 0.0, "entity_diversity": 0.0}
        
        # Completeness: ratio of nodes with type information
        typed_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('type') != 'UNKNOWN')
        completeness = typed_nodes / G.number_of_nodes()
        
        # Connectivity: edge-to-node ratio
        connectivity = G.number_of_edges() / max(G.number_of_nodes(), 1)
        
        # Entity diversity: number of unique entity types
        entity_types = set(nx.get_node_attributes(G, 'type').values())
        entity_diversity = len(entity_types) / max(G.number_of_nodes(), 1)
        
        return {
            "completeness": completeness,
            "connectivity": connectivity,
            "entity_diversity": entity_diversity
        }

    def build_and_evaluate(self, entities: Dict[str, List[str]], 
                          relations: List[Dict[str, str]], 
                          name: str,
                          save_visualization: bool = True) -> Dict[str, Any]:
        """Build graph, calculate statistics, and save outputs in one pipeline"""
        try:
            # Build graph with validation
            G = self.build_graph(entities, relations)
            
            # Calculate comprehensive statistics
            stats = self.get_statistics(G)
            
            # Save JSON representation
            json_path = self.save_graph(G, name)
            
            # Optionally create visualization
            viz_path = ""
            if save_visualization and G.number_of_nodes() > 0:
                try:
                    viz_path = self.visualize(G, name)
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            # ML pipeline integration
            result = {
                "graph": G,
                "statistics": stats,
                "paths": {
                    "json": json_path,
                    "visualization": viz_path
                },
                "data_quality": self._assess_data_quality(G)
            }
            
            return result
            
        except Exception as e:
            print(f"Error: Failed to build graph '{name}': {e}")
            raise

    def load_graph(self, path: str) -> nx.DiGraph:
        """Load a graph from a JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        G = nx.DiGraph()
        for node in data.get('nodes', []): # Added .get for safety
            G.add_node(node['id'], type=node.get('type', 'UNKNOWN'))
        for edge in data.get('edges', []): # Added .get for safety
            G.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'RELATED'))
            
        return G
