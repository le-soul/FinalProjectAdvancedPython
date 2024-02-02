"""

"""

from scripts.linearregression import PredictClass
import pandas as pd
import pytest
from sklearn.datasets import make_regression

class Test:

    @pytest.fixture
    def example_data(self):
        # Generate example data
        X, y = make_regression(n_samples=100, n_features=6, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['sq_mt_built', 'n_rooms', 'n_bathrooms', 'district_id', 'has_parking', 'has_ac'])
        df['buy_price'] = y
        return df

    def test_price_as_y(self, example_data):
        predictor = PredictClass(example_data)
        # No assertion as this method plots data

    def test_multicollinearity_and_model_equation(self, example_data):
        predictor = PredictClass(example_data)
        predictor.multicollinearity_and_model_equation()
        

    def test_predict_price(self, monkeypatch):
        input_data = [
            "100",  # sq_mt_built
            "3",    # n_rooms
            "2",    # n_bathrooms
            "1",    # district_id
            "1",    # has_parking
            "1"     # has_ac
        ]
        # Mocking user input
        monkeypatch.setattr('builtins.input', lambda _: input_data.pop(0))
        
        predictor = PredictClass(None)
        # No assertion as this method prints the predicted price
