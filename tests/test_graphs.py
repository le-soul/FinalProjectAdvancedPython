"""
Defines test cases for visualizing data related to house prices and characteristics in Madrid.
"""

import unittest
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from scripts.graphs import ViewClass

matplotlib.use("Agg")

class TestViewClass(unittest.TestCase):
    """
    A class for testing the ViewClass methods.
    """

    def setUp(self):
        """
        Set up a sample DataFrame and ViewClass object for testing.
        """
        self.df = pd.DataFrame(
            {
                "buy_price": [100, 200, 300],
                "district_id": [1, 2, 1],
                "n_rooms": [2, 3, 4],
                "n_bathrooms": [1, 2, 2],
                "built_year": [2000, 2010, 2020],
                "sq_mt_built": [50, 80, 100],
            }
        )
        self.view_obj = ViewClass(self.df)

    def test_correlation_matrix(self):
        """
        Test the correlation matrix function.
        """
        corrmat = self.view_obj.correlation_matrix()
        if corrmat is not None:
            self.assertIsInstance(corrmat, pd.DataFrame)
            self.assertEqual(corrmat.shape, (6, 6))
            self.assertTrue(
                corrmat.loc["buy_price", "sq_mt_built"] > 0
            ) 
            self.assertAlmostEqual(
                corrmat.loc["district_id", "buy_price"], 0, places=4
            )
            self.assertTrue((corrmat == corrmat.T).all().all())
            self.assertTrue(
                (corrmat.values.diagonal() == 1).all()
            )

        else:
            print("Error: correlation_matrix returned None")

        plt.close()

    def test_price_skewness(self):
        """
        Test the price skewness function.
        """
        skewness = self.view_obj.price_skewness()
        if skewness is not None:
            self.assertIsInstance(skewness, float)
            expected_skewness = self.df["buy_price"].skew()
            self.assertAlmostEqual(skewness, expected_skewness, places=4)
        else:
            print("Error: price_skewness returned None")

    def test_most_exp_districts(self):
        """
        Test the function for finding the most expensive districts.
        """
        self.view_obj.most_exp_districts()
        self.assertEqual(
            plt.gca().get_title(), "Top 21 Expensive Districts by Average Buy Price"
        )
        district_label = plt.gca().get_xlabel()
        self.assertEqual(district_label, "District ID")
        self.assertEqual(plt.gca().get_ylabel(), "Average Buy Price")
        plt.close()

    def test_most_bathrooms_districts(self):
        """
        Test the function for finding the districts with the most bathrooms.
        """
        self.view_obj.most_bathrooms_districts()
        self.assertEqual(
            plt.gca().get_title(), "Average Number of Bathrooms by District"
        )
        district_label = plt.gca().get_xlabel()
        self.assertEqual(district_label, "District ID")
        self.assertEqual(plt.gca().get_ylabel(), "Average Number of Bathrooms")
        plt.close()

    def test_most_rooms_districts(self):
        """
        Test the function for finding the districts with the most rooms.
        """
        self.view_obj.most_rooms_districts()
        self.assertEqual(plt.gca().get_title(), "Average Number of Rooms by District")
        district_label = plt.gca().get_xlabel()
        self.assertEqual(district_label, "District ID")
        self.assertEqual(plt.gca().get_ylabel(), "Average Number of Rooms")
        plt.close()

    def test_price_and(self):
        """
        Test the function for plotting price against various features.
        """
        self.view_obj.price_and()
        self.assertGreater(
            len(plt.gcf().get_axes()), 0, "Error: price_and did not generate any plot"
        )
        plt.close()


if __name__ == "__main__":
    unittest.main()
