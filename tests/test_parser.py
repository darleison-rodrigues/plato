from contexter.parser import PDFProcessor
import os

def test_batch_processing():
    processor = PDFProcessor()
    
    # Test with a non-existent file to check fault tolerance
    files = ["non_existent_1.pdf", "non_existent_2.pdf"]
    
    print("Testing batch_convert with missing files (should print errors but not crash)...")
    results = processor.batch_convert(files)
    
    print(f"Processed {len(results)} files successfully.")
    
    if len(results) == 0:
        print("Success: Batch processing handled errors gracefully.")
    else:
        print("Failure: Unexpected results.")

if __name__ == "__main__":
    test_batch_processing()
