"""
Test loading of the dataset
"""

import os
import unittest
import pandas as pd
from scripts.main import load_dataset


class LoadDatasetTest(unittest.TestCase):

    """
    Test cases for the load_dataset function.
    """

    def test_load_dataset_not_csv(self):
        """
        Test loading a file with an invalid extension.
        """

        with self.assertRaises(TypeError):
            load_dataset("test.txt")

    def test_load_dataset_file_not_found(self):
        """
        Test loading a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            load_dataset("non_existent_file.csv")

    def test_load_dataset_empty_csv(self):
        """
        Test loading an empty CSV file.
        """
        with open("dataset/empty.csv", "w"):
            pass
        with self.assertRaises(ValueError):
            load_dataset("dataset/empty.csv")
        os.remove("dataset/empty.csv")


if __name__ == "__main__":
    unittest.main()
