import contexter.patch
import pytest
import chromadb.config

def test_chroma_config_init():
    """Verify ChromaDB settings execute without crashing on Python 3.14"""
    try:
        # Just accessing the class triggers the metaclass logic that fails
        _ = chromadb.config.Settings()
    except Exception as e:
        pytest.fail(f"ChromaDB Settings initialization failed: {e}")

if __name__ == "__main__":
    test_chroma_config_init()
    print("Success: ChromaDB settings initialized.")
