from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

class GraphVisualizer:
    """Generate graph diagrams from entity/relation data"""
    
    @staticmethod
    def generate_mermaid(
        entities: List[dict],
        relations: List[dict],
        max_nodes: int = 50
    ) -> str:
        """
        Generate Mermaid diagram syntax.
        Docs: https://mermaid.js.org/syntax/flowchart.html
        """
        lines = ["graph TD"]
        
        # Limit to most connected entities if too many
        if len(entities) > max_nodes:
            # Count connections per entity
            entity_counts = {}
            for rel in relations:
                entity_counts[rel['source']] = entity_counts.get(rel['source'], 0) + 1
                entity_counts[rel['target']] = entity_counts.get(rel['target'], 0) + 1
            
            # Keep top N
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            entity_names = {e[0] for e in top_entities}
            
            entities = [e for e in entities if e['name'] in entity_names]
            relations = [r for r in relations if r['source'] in entity_names and r['target'] in entity_names]
        
        # Create node IDs (sanitize names for Mermaid)
        node_map = {}
        for i, entity in enumerate(entities):
            node_id = f"node{i}"
            node_map[entity['name']] = node_id
            
            # Style by type
            shape = GraphVisualizer._get_mermaid_shape(entity['type'])
            # sanitize name for display
            display_name = entity['name'].replace('"', '').replace('(', '').replace(')', '')
            lines.append(f"    {node_id}{shape}".replace('name', display_name))
        
        # Add edges
        for rel in relations:
            source_id = node_map.get(rel['source'])
            target_id = node_map.get(rel['target'])
            
            if source_id and target_id:
                relation_label = rel['relation'].replace('_', ' ').replace('"', '')
                lines.append(f"    {source_id} -->|{relation_label}| {target_id}")
        
        # Add styling
        lines.extend([
            "",
            "    classDef person fill:#e1f5ff,stroke:#01579b",
            "    classDef org fill:#fff3e0,stroke:#e65100",
            "    classDef concept fill:#f3e5f5,stroke:#4a148c",
        ])
        
        # Apply styles
        for ent_type in ['PERSON', 'ORGANIZATION', 'CONCEPT']:
            # Handle variations in type naming
            nodes = []
            for e in entities:
                t = e['type'].upper()
                if t == ent_type or (ent_type == 'ORGANIZATION' and t in ['ORG', 'ORGANISATION']):
                    if e['name'] in node_map:
                        nodes.append(node_map[e['name']])
                        
            if nodes:
                class_name = ent_type.lower().replace('organization', 'org')
                lines.append(f"    class {','.join(nodes)} {class_name}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _get_mermaid_shape(entity_type: str) -> str:
        """Get Mermaid shape based on entity type"""
        entity_type = entity_type.upper()
        shapes = {
            'PERSON': '([name])',           # Stadium shape
            'ORGANIZATION': '[name]',       # Rectangle
            'ORG': '[name]',                # Rectangle
            'CONCEPT': '{{name}}',          # Hexagon
            'LOCATION': '[(name)]',         # Cylinder
            'EVENT': '>name]',              # Asymmetric
        }
        return shapes.get(entity_type, '[name]')
    
    @staticmethod
    def generate_graphviz(
        entities: List[dict],
        relations: List[dict]
    ) -> str:
        """
        Generate GraphViz DOT syntax.
        More control than Mermaid, requires graphviz installation.
        """
        lines = [
            "digraph KnowledgeGraph {",
            "    rankdir=LR;",
            "    node [style=filled];",
            ""
        ]
        
        # Add nodes
        for i, entity in enumerate(entities):
            color = GraphVisualizer._get_graphviz_color(entity['type'])
            shape = GraphVisualizer._get_graphviz_shape(entity['type'])
            label = entity['name'].replace('"', '\\"')
            lines.append(
                f'    node{i} [label="{label}", fillcolor="{color}", shape={shape}];'
            )
        
        # Add edges
        node_map = {e['name']: f"node{i}" for i, e in enumerate(entities)}
        for rel in relations:
            source = node_map.get(rel['source'])
            target = node_map.get(rel['target'])
            if source and target:
                label = rel['relation'].replace('_', ' ').replace('"', '\\"')
                lines.append(f'    {source} -> {target} [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    @staticmethod
    def _get_graphviz_color(entity_type: str) -> str:
        entity_type = entity_type.upper()
        colors = {
            'PERSON': 'lightblue',
            'ORGANIZATION': 'lightyellow',
            'ORG': 'lightyellow',
            'CONCEPT': 'lightgreen',
            'LOCATION': 'lightpink',
        }
        return colors.get(entity_type, 'white')
    
    @staticmethod
    def _get_graphviz_shape(entity_type: str) -> str:
        entity_type = entity_type.upper()
        shapes = {
            'PERSON': 'ellipse',
            'ORGANIZATION': 'box',
            'ORG': 'box',
            'CONCEPT': 'diamond',
            'LOCATION': 'house',
        }
        return shapes.get(entity_type, 'box')
    
    @staticmethod
    def wrap_mermaid_html(mermaid_code: str, title: str = "Knowledge Graph") -> str:
        """Wrap Mermaid code in standalone HTML file"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }}
    </style>
</head>
<body>
    <h1>🦫 {title}</h1>
    <div class="info">
        <strong>Interactive Graph:</strong> Pan, zoom, and explore your knowledge graph.
        <br><strong>Generated by:</strong> Plato
    </div>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ 
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>"""
