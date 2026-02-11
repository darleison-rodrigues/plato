import os
import numpy as np
from plato.core.retriever import VectorRetriever
from plato.models.document import SearchResult

def mock_embedding_fn(texts):
    import hashlib
    res = []
    for t in texts:
        # Create a unique but deterministic vector
        h = hashlib.sha256(t.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Pad to 768
        padded = np.zeros(768, dtype=np.float32)
        padded[:len(vec)] = vec
        res.append(padded.tolist())
    return res

def test_native_retriever():
    persist_dir = "./test_vectors"
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        
    retriever = VectorRetriever(
        persist_dir=persist_dir,
        embedding_fn=mock_embedding_fn
    )
    
    # 1. Test Indexing
    print("Testing indexing...")
    retriever.index_batch(
        ["doc1", "doc2"],
        ["This is doc 1", "This is document two"],
        [
            {"pdf_hash": "hash_12345678", "page": 1, "chunk_index": 0}, 
            {"pdf_hash": "hash_12345678", "page": 1, "chunk_index": 1}
        ]
    )
    
    stats = retriever.get_stats()
    print(f"Stats: {stats}")
    assert stats["total_chunks"] == 2
    
    # 2. Test Query
    print("Testing query...")
    results = retriever.query("This is doc 1", n_results=1)
    assert len(results) == 1
    assert results[0].doc_id == "doc1"
    print(f"Query Result: {results[0].doc_id} with distance {results[0].distance}")
    
    # 3. Test Persistence
    print("Testing persistence...")
    del retriever
    retriever_new = VectorRetriever(
        persist_dir=persist_dir,
        embedding_fn=mock_embedding_fn
    )
    assert retriever_new.get_stats()["total_chunks"] == 2
    print("Persistence verified.")
    
    # 4. Test Deletion
    print("Testing deletion...")
    # Add a mock pdf_hash
    retriever_new.index_document(
        "doc3", 
        "doc 3", 
        {"pdf_hash": "hash12345678", "page": 2, "chunk_index": 0}
    )
    count = retriever_new.delete_pdf("hash12345678")
    assert count == 1
    assert retriever_new.get_stats()["total_chunks"] == 2
    print("Deletion verified.")
    
    print("\n✅ Native VectorRetriever test PASSED!")

if __name__ == "__main__":
    test_native_retriever()
