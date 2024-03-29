"""
Class for predictions of classification data
"""
# pylint:disable=C0103
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DecisionClass:
    """
    Class for predicting using decision trees
    """

    def __init__(self, df):
        self.df = df

    def plot_decision_tree(self, tree_model, input_features, fontsize=12):
        """
        Plot the decision tree generated by the provided model
        """

        _, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            tree_model,
            filled=True,
            impurity=False,
            feature_names=input_features,
            class_names=["No", "Yes"],
            proportion=True,
            ax=ax,
            fontsize=fontsize,
        )

    def decision_tree_parking(self):
        """
        Perform decision tree classification to predict parking availability based on price and district_id
        """

        input_features = ["buy_price", "district_id"]
        X = self.df[input_features]
        y = self.df["has_parking"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=1
        )

        tree_model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

        self.plot_decision_tree(tree_model, input_features)

        y_pred = tree_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy: ", round(100 * accuracy, 1), "%")
        print("Precision: ", round(100 * precision, 1), "%")
        print("Recall: ", round(100 * recall, 1), "%")
        print("F1 score: ", round(100 * f1, 1), "%")

        return accuracy, precision, recall, f1

    def decision_tree_ac(self):
        """
        Perform decision tree classification to predict parking availability if houses have ac based on price
        """

        input_features = ["buy_price"]
        X = self.df[input_features]
        y = self.df["has_ac"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=1
        )

        tree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

        self.plot_decision_tree(tree_model, input_features)

        y_pred = tree_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy: ", round(100 * accuracy, 1), "%")
        print("Precision: ", round(100 * precision, 1), "%")
        print("Recall: ", round(100 * recall, 1), "%")
        print("F1 score: ", round(100 * f1, 1), "%")

        return accuracy, precision, recall, f1
