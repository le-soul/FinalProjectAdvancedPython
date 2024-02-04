"""
Script for predictions
"""
# pylint:disable=C0103
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


class PredictClass:
    """
    Class for creating a linear regression
    """

    def __init__(self, df):
        """
        Initialize PredictClass with dataframe.
        """
        self.df = df
        self.df1 = None

    def price_as_y(self):
        """
        Perform linear regression on buy price.
        """
        X = self.df[["sq_mt_built", "n_rooms", "has_parking", "has_ac"]]
        y = self.df["buy_price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        plt.scatter(X_test["sq_mt_built"], y_test, color="black", label="Actual")
        plt.scatter(X_test["sq_mt_built"], y_pred, color="blue", label="Predicted")
        plt.xlabel("Square Meters Built")
        plt.ylabel("Buy Price")
        plt.legend()
        plt.title("Linear Regression: Actual vs. Predicted Buy Price")
        return y_test, y_pred

    def log_price_as_y(self):
        """
        Perform linear regression on log buy price.
        """
        self.df1 = self.df.copy()
        log_price = np.log(self.df1["buy_price"])
        log_sq_mt_built = np.log(self.df1["sq_mt_built"])
        self.df1["log_price"] = log_price
        self.df1["log_sq_mt_built"] = log_sq_mt_built

        X = self.df1[
            ["log_sq_mt_built", "n_rooms", "has_parking", "has_ac"]
        ]
        y = self.df1["log_price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        plt.scatter(X_test["log_sq_mt_built"], y_test, color="black", label="Actual")
        plt.scatter(X_test["log_sq_mt_built"], y_pred, color="blue", label="Predicted")
        plt.xlabel("Log of Square Meters Built")
        plt.ylabel("Log of Buy Price")
        plt.legend()
        plt.title("Linear Regression: Actual vs. Predicted Log of Buy Price")

        r_squared = r2_score(y_test, y_pred)
        print(f"\nR-squared: {r_squared}")

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        sk_model = LinearRegression()
        sk_model.fit(X_train, y_train)

        y_pred = sk_model.predict(X_test)

        X_with_const = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ]
        print(f"\nMulticollinearity test(VIF)\n {vif_data}")

    def multicollinearity_and_model_equation(self):
        """
        Shows the multicollinearity, the R squared of the linear regression and creates a model for use
        """

        X = self.df[["sq_mt_built", "n_rooms", "has_parking", "has_ac"]]
        y = self.df["buy_price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        sk_model = LinearRegression()
        sk_model.fit(X_train, y_train)

        y_pred = sk_model.predict(X_test)

        X_with_const = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ]

        print(f"\nMulticollinearity test(VIF)\n {vif_data}")

        r_squared = r2_score(y_test, y_pred)
        print(f"\nR-squared: {r_squared}")

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        X = sm.add_constant(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        sts_train = sm.OLS(y_train, X_train)
        result = sts_train.fit()

        print("\n", result.summary())

        params = result.params
        intercept = params["const"]
        coefficients = params.drop("const")

        print("\nMultiple regression model:\nBuy Price =", end=" ")

        for i, (column_name, coefficient) in enumerate(coefficients.items()):
            coefficient_rounded = round(coefficient, 2)
            print(f"{coefficient_rounded} * {column_name}", end="")
            if i < len(coefficients) - 1:
                print(" + ", end="")
        print(f" + {round(intercept, 2)}")
        return vif_data, r_squared, rmse, result.summary()

    def predict_price(self):
        """
        Predict buy price based on user input.
        """
        try:
            beta_1 = 4605.19
            beta_2 = -3637.74
            beta_3 = -14343.35
            beta_4 = 69065.09
            intercept = -60071.3

            sq_mt_built = float(input("Enter square meters built: "))
            n_rooms = float(input("Enter number of rooms: "))
            has_parking = int(input("Enter 1 if there is parking, 0 otherwise: "))
            has_ac = int(input("Enter 1 if there is AC, 0 otherwise: "))

            predicted_price = (
                beta_1 * sq_mt_built 
                + beta_2 * n_rooms 
                + beta_3 * has_parking 
                + beta_4 * has_ac 
                + intercept
            )
            predicted_price = round(predicted_price, 2)

            print(predicted_price)
        except ValueError as ve:
            print(f"Error: {ve}. Please enter a valid numeric value.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
