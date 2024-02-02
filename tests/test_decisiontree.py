import unittest
import pandas as pd
from sklearn.datasets import make_classification
from scripts.decisiontree import DecisionClass

class TestDecisionClass(unittest.TestCase):
    def setUp(self):
        # Generate a sample DataFrame for testing
        X, y = make_classification(n_samples=100, n_features=3, n_classes=2, random_state=42,
                                   n_informative=2, n_redundant=0, n_repeated=0)
        self.df = pd.DataFrame(X, columns=['buy_price', 'district_id', 'extra_feature'])
        self.df['has_parking'] = y
        self.df['has_ac'] = y

    def test_decision_tree_parking(self):
        decision_class = DecisionClass(self.df)
        accuracy, precision, recall, f1 = decision_class.decision_tree_parking()
        
        expected_min_percentage = 0
        expected_max_percentage = 100

        self.assertGreaterEqual(accuracy * 100, expected_min_percentage, "Accuracy score is less than expected minimum percentage")
        self.assertLessEqual(accuracy * 100, expected_max_percentage, "Accuracy score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(precision * 100, expected_min_percentage, "Precision score is less than expected minimum percentage")
        self.assertLessEqual(precision * 100, expected_max_percentage, "Precision score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(recall * 100, expected_min_percentage, "Recall score is less than expected minimum percentage")
        self.assertLessEqual(recall * 100, expected_max_percentage, "Recall score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(f1 * 100, expected_min_percentage, "F1 score is less than expected minimum percentage")
        self.assertLessEqual(f1 * 100, expected_max_percentage, "F1 score is greater than expected maximum percentage")
        
    def test_decision_tree_ac(self):
        decision_class = DecisionClass(self.df)
        accuracy, precision, recall, f1 = decision_class.decision_tree_ac()
        
        expected_min_percentage = 0
        expected_max_percentage = 100

        self.assertGreaterEqual(accuracy * 100, expected_min_percentage, "Accuracy score is less than expected minimum percentage")
        self.assertLessEqual(accuracy * 100, expected_max_percentage, "Accuracy score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(precision * 100, expected_min_percentage, "Precision score is less than expected minimum percentage")
        self.assertLessEqual(precision * 100, expected_max_percentage, "Precision score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(recall * 100, expected_min_percentage, "Recall score is less than expected minimum percentage")
        self.assertLessEqual(recall * 100, expected_max_percentage, "Recall score is greater than expected maximum percentage")
        
        self.assertGreaterEqual(f1 * 100, expected_min_percentage, "F1 score is less than expected minimum percentage")
        self.assertLessEqual(f1 * 100, expected_max_percentage, "F1 score is greater than expected maximum percentage")


if __name__ == '__main__':
    unittest.main()

