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
        Shows the multicollinearity, the R squared of the linear regression and creates a model for use
        """
        
        X = self.df[['sq_mt_built', 'n_rooms', 'n_bathrooms', 'district_id' ,'has_parking', 'has_ac']]
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
        print(f'\nMulticollinearity test(VIF)\n {vif_data}')

        r_squared = r2_score(y_test, y_pred)
        print(f'\nR-squared: {r_squared}')

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        X = sm.add_constant(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and fit the multiple linear regression model
        model = sm.OLS(y_train, X_train)
        result = model.fit()

        print("\n", result.summary())

        # Get the coefficients and intercept
        params = result.params
        intercept = params['const']
        coefficients = params.drop('const')

        print("\nMultiple regression model:\nBuy Price =", end=" ")
        for i in range(len(coefficients)):
            coefficient_rounded = round(coefficients.iloc[i], 2)
            print(f"{coefficient_rounded} * {X.columns[i]}", end="")
            if i < len(coefficients) - 1:
                print(" + ", end="")
        print(f" + {round(intercept, 2)}")

    def predict_price(self):
        try:
            # Get user input for independent variables
            sq_mt_built = float(input("Enter square meters built: "))
            n_rooms = float(input("Enter number of rooms: "))
            n_bathrooms = float(input("Enter number of bathrooms: "))
            district_id = float(input("District 1: Arganzuela, District 2: Barajas, District 3: Carabanchel, District 4: Centro,"
                                    "District 5: Chamartín, District 6: Chamberí, District 7: Ciudad Lineal, District 8: Fuencarral, District 9: Hortaleza,"
                                    "District 10: Latina, District 11: Moncloa, District 12: Moratalaz, District 13: Puente de Vallecas, District 14: Retiro,"
                                    "District 15: Salamanca, District 17: Tetuán, District 18: Usera, District 19: Vicálvaro, District 20: Villa de Vallecas,"
                                    "District 21: Villaverde\nEnter district ID: "))
            has_parking = int(input("Enter 1 if there is parking, 0 otherwise: "))
            has_ac = int(input("Enter 1 if there is AC, 0 otherwise: "))

            # Calculate predicted price using coefficients and intercept
            predicted_price = 3869.81 * sq_mt_built + -22704.11 * n_rooms + 105741.4 * n_bathrooms + \
                            -8381.5 * district_id + -32317.3 * has_parking + 45670.57 * has_ac + -12905.74

            predicted_price = round(predicted_price, 2)

            print(predicted_price)
        except ValueError as ve:
            print(f"Error: {ve}. Please enter a valid numeric value.")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")

