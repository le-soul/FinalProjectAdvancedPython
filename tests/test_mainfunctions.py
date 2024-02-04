"""
Defines test cases for the options in the command line
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from scripts.mainfunctions import MainClass


class TestPlotter(unittest.TestCase):
    """
    Test cases for the plotter functionality.
    """

    @patch("scripts.mainfunctions.plt")
    @patch("builtins.print")
    def test_plotter(self, print_mock, plt_mock):
        """
        Test the plotter method.
        """

        viewer_mock = MagicMock()

        MainClass.plotter(True, viewer_mock, "correlation", "outputs")

        viewer_mock.correlation_matrix.assert_called_once()
        plt_mock.savefig.assert_called_once_with(
            os.path.join("outputs", "correlation_matrix.png")
        )
        print_mock.assert_called_once_with(
            f"Graph saved at: {os.path.join('outputs', 'correlation_matrix.png')}"
        )

    @patch("scripts.mainfunctions.plt")
    @patch("builtins.print")
    def test_regression(self, print_mock, plt_mock):
        """
        Test the regression method.
        """
        trainer_mock = MagicMock()

        MainClass.regression(True, "regression", trainer_mock, "outputs")

        trainer_mock.price_as_y.assert_called_once()
        plt_mock.savefig.assert_called_once_with(
            os.path.join("outputs", "Linear_Regression.png")
        )
        print_mock.assert_called_once_with(
            f"Graph saved at: {os.path.join('outputs', 'Linear_Regression.png')}"
        )

    @patch("scripts.mainfunctions.plt")
    def test_classifier(self, plt_mock):
        """
        Test the classifier method.
        """

        tree_mock = MagicMock()

        MainClass.classifier(True, "has ac", tree_mock, "outputs")

        self.assertTrue(tree_mock.decision_tree_ac.called)
        call_args_list = plt_mock.savefig.call_args_list
        saved_files = [call[0][0] for call in call_args_list]
        self.assertIn(os.path.join("outputs", "Decision_Tree_Ac.png"), saved_files)


if __name__ == "__main__":
    unittest.main()
