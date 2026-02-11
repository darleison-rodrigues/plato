import asyncio
import numpy as np

def run_numpy():
    print("Start numpy in thread")
    # Simulate the exact ops in retriever
    vectors = np.random.rand(1, 768).astype(np.float32)
    query = np.random.rand(768).astype(np.float32)
    
    print("Computing vector norms")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / (norms + 1e-9)
    print("Vectors normalized")
    
    print("Computing query norm")
    q_norm = np.linalg.norm(query)
    normalized_q = query / (q_norm + 1e-9)
    print("Query normalized")
    
    print("Computing dot")
    sims = np.dot(normalized_vectors, normalized_q)
    print(f"Dot result: {sims}")

async def main():
    print("Running in main asyncio loop")
    await asyncio.to_thread(run_numpy)
    print("Done")

if __name__ == "__main__":
    asyncio.run(main())
