"""
Script for predictions
"""

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
        self.df = df

    def price_as_y(self):
        # Adding has_storage_room increases the R squared very slightly and makes the model more complex so I rather not include it.
        X = self.df[['sq_mt_built', 'n_rooms', 'n_bathrooms', 'district_id' ,'has_parking', 'has_ac']]
        y = self.df['buy_price']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a linear regression model and fit it to the training data
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Plot the actual vs. predicted values
        plt.scatter(X_test['sq_mt_built'], y_test, color='black', label='Actual')
        plt.scatter(X_test['sq_mt_built'], y_pred, color='blue', label='Predicted')
        plt.xlabel('Square Meters Built')
        plt.ylabel('Buy Price')
        plt.legend()
        plt.title('Linear Regression: Actual vs. Predicted Buy Price')

        

    def multicollinearity_and_model_equation(self):
        """
        aaaaaaaaaaaaaaaaaaaaaa
        """

        
        X = self.df[['sq_mt_built', 'n_bathrooms', 'district_id' ,'has_parking', 'has_ac']]
        y = self.df['buy_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Add a constant term to the features matrix (required for statsmodels)
        X_with_const = sm.add_constant(X)

        # Create and fit the linear regression model
        model = sm.OLS(y, X_with_const).fit()

        # Calculate VIF for each variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

        # Display the VIF values
        print(vif_data)

        r_squared = r2_score(y_test, y_pred)
        print(f'R-squared: {r_squared}')

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        # Get the coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_

        print("Buy Price =", end=" ")
        for i in range(len(coefficients)):
            coefficient_rounded = round(coefficients[i], 2)
            print(f"{coefficient_rounded} * {X.columns[i]}", end="")
            if i < len(coefficients) - 1:
                print(" + ", end="")
        print(f" + {round(intercept, 2)}") 