import unittest
from unittest.mock import patch
from pathlib import Path
import sys

# Adjust path to find local modules
sys.path.append(str(Path(__file__).parent.parent))
from plato.parser import DocumentParser

class TestDocumentParser(unittest.TestCase):



    def setUp(self):

        self.parser = DocumentParser()



        @patch("plato.parser.SimpleDirectoryReader")



        def test_load_data_success(self, mock_reader):



            """Test that load_data calls the underlying reader successfully."""



            mock_reader.return_value.load_data.return_value = ["doc1"]



            



            dummy_dir = Path("./fake_dir")



            # We need to mock the is_dir check that happens inside the method



            with patch("pathlib.Path.is_dir", return_value=True):



                documents = self.parser.load_data(dummy_dir)



            



            self.assertEqual(len(documents), 1)



            mock_reader.assert_called_with(input_dir=str(dummy_dir), recursive=True)



    def test_load_data_invalid_dir(self):

        """Test that load_data raises an error for a non-directory."""

        with patch("pathlib.Path.is_dir", return_value=False):

            with self.assertRaises(ValueError):

                self.parser.load_data(Path("./not_a_directory.txt"))

if __name__ == "__main__":
    unittest.main()
