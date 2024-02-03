"""
Defines test cases for linear regressions
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from scripts.linearregression import PredictClass


class TestPredictClass:
    """
    Test cases for the PredictClass class.
    """

    @pytest.fixture
    def example_data(self):
        """
        Generate example data
        """
        X, y = make_regression(n_samples=100, n_features=6, noise=0.1, random_state=42)
        df = pd.DataFrame(
            X,
            columns=[
                "sq_mt_built",
                "n_rooms",
                "n_bathrooms",
                "district_id",
                "has_parking",
                "has_ac",
            ],
        )
        df["buy_price"] = y
        return df

    def test_price_as_y(self, example_data):
        """
        Test the price_as_y method.
        """
        predictor = PredictClass(example_data)
        y_test, y_pred = predictor.price_as_y()
        assert isinstance(y_test.values, np.ndarray)
        # Add more assertions as needed

    def test_log_price_as_y(self, example_data):
        """
        Test the log_price_as_y method.
        """
        predictor = PredictClass(example_data)
        # Add test logic here

    def test_multicollinearity_and_model_equation(self, example_data):
        """
        Test the multicollinearity_and_model_equation method
        """
        predictor = PredictClass(example_data)
        # Add test logic here

    def test_predict_price(self, monkeypatch):
        """
        Test the predict_price method.
        """
        input_data = [
            "100",  # sq_mt_built
            "3",  # n_rooms
            "1",  # district_id
            "1",  # has_parking
            "1",  # has_ac
        ]
        # Mocking user input
        monkeypatch.setattr("builtins.input", lambda _: input_data.pop(0))

        predictor = PredictClass(None)
        # Add test logic here
