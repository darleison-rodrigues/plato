from plato.llm import OllamaClient
import unittest
from unittest.mock import MagicMock, patch

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def test_extract_json_valid(self):
        """Test extraction of valid JSON from markdown"""
        text = "Here is the data: ```json\n{\"entities\": {\"PERSON\": [\"Alice\"]}}\n```"
        result = self.client._extract_json(text)
        self.assertEqual(result, {"entities": {"PERSON": ["Alice"]}})

    def test_extract_json_raw(self):
        """Test extraction of raw JSON"""
        text = "Some text {\"key\": \"value\"} more text."
        result = self.client._extract_json(text)
        self.assertEqual(result, {"key": "value"})
        
    # def test_metrics_calculation(self):
    #     """Test validation metrics calculation"""
    #     # Perfect match
    #     results = [{
    #         'extracted': {'entities': {'PERSON': ['Alice']}},
    #         'ground_truth': {'entities': {'PERSON': ['Alice']}},
    #         'extraction_time': 0.1
    #     }]
    #     metrics = self.client._calculate_metrics(results)
    #     self.assertEqual(metrics['precision'], 1.0)
    #     self.assertEqual(metrics['recall'], 1.0)
    #     self.assertEqual(metrics['f1'], 1.0)
    #     
    #     # Partial match
    #     results = [{
    #         'extracted': {'entities': {'PERSON': ['Alice', 'Bob']}}, # Bob is False Positive
    #         'ground_truth': {'entities': {'PERSON': ['Alice', 'Charlie']}}, # Charlie is False Negative
    #         'extraction_time': 0.1
    #     }]
    #     metrics = self.client._calculate_metrics(results)
    #     # Precision: 1/2 = 0.5
    #     # Recall: 1/2 = 0.5
    #     self.assertEqual(metrics['precision'], 0.5)
    #     self.assertEqual(metrics['recall'], 0.5)

if __name__ == '__main__':
    unittest.main()
