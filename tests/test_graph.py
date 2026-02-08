from contexter.graph import KnowledgeGraphBuilder
import networkx as nx
import os

def test_graph_builder():
    builder = KnowledgeGraphBuilder()
    
    # Test 1: Validation (Empty Inputs)
    print("Test 1: Testing Validation (Empty Inputs)...")
    try:
        builder.build_graph({}, [])
        print("Failure: Should have raised ValueError")
    except ValueError as e:
        print(f"Success: Caught expected error: {e}")

    # Test 2: Build & Evaluate Pipeline
    print("\nTest 2: Testing Build & Evaluate Pipeline...")
    entities = {"PERSON": ["Alice", "Bob"], "ORG": ["Corp"]}
    relations = [{"subject": "Alice", "relation": "WORKS_FOR", "object": "Corp"}]
    
    try:
        result = builder.build_and_evaluate(entities, relations, "test_graph", save_visualization=True)
        
        # Check components
        assert isinstance(result['graph'], nx.DiGraph)
        assert "completeness" in result['data_quality']
        assert os.path.exists(result['paths']['json'])
        # Visualization might be skipped in some envs, but path key should exist
        assert "visualization" in result['paths']
        
        print("Success: Pipeline ran and produced metrics.")
        print("Data Quality Stats:", result['data_quality'])
        
    except Exception as e:
        print(f"Failure: Pipeline crashed: {e}")

if __name__ == "__main__":
    test_graph_builder()
