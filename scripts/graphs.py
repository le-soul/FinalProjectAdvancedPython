"""
Class made for the view of data
"""

import seaborn as sns
import matplotlib.pyplot as plt


class ViewClass:
    """
    Class for viewing data gathered in a functional way
    """

    def __init__(self, df):
        self.df = df

    def correlation_matrix(self):
        """
        Function to see the correlation between variables in the database
        """

        corrmat = self.df.corr()
        f, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corrmat, vmax=0.8, square=True)
        return self.df.corr()

    def price_skewness(self):
        """
        Function to see the skewness in the buying price of houses in Madrid
        """

        sns.histplot(
            self.df["buy_price"],
            kde=True,
            stat="density",
            kde_kws=dict(cut=3),
            alpha=0.4,
            edgecolor=(1, 1, 1, 0.4),
            bins=40,
        )
        print(f"Skewness: {self.df['buy_price'].skew()}")
        return self.df["buy_price"].skew()

    def most_exp_districts(self):
        """
        Function to see the most expensive districts in Madrid in average plotted
        """

        top_21_expensive_districts = (
            self.df.groupby("district_id")["buy_price"].mean().nlargest(21)
        )

        plt.figure(figsize=(10, 6))
        top_21_expensive_districts.plot(kind="bar", color="skyblue")
        plt.title("Top 21 Expensive Districts by Average Buy Price")
        plt.xlabel("District ID")
        plt.ylabel("Average Buy Price")
        plt.xticks(rotation=45)
        plt.tight_layout()

    def most_rooms_districts(self):
        """
        Function to see where bigger houses are most commonly located plotted
        """

        average_rooms_per_district = (
            self.df.groupby("district_id")["n_rooms"].mean().nlargest(21)
        )

        plt.figure(figsize=(10, 6))
        average_rooms_per_district.plot(kind="bar", color="lightgreen")
        plt.title("Average Number of Rooms by District")
        plt.xlabel("District ID")
        plt.ylabel("Average Number of Rooms")
        plt.xticks(rotation=45)
        plt.tight_layout()

    def most_bathrooms_districts(self):
        """
        Function see the number of bathrooms per district in average plotted
        """

        average_rooms_per_district = (
            self.df.groupby("district_id")["n_bathrooms"].mean().nlargest(21)
        )

        plt.figure(figsize=(10, 6))
        average_rooms_per_district.plot(kind="bar", color="#ffcccb")
        plt.title("Average Number of Bathrooms by District")
        plt.xlabel("District ID")
        plt.ylabel("Average Number of Bathrooms")
        plt.xticks(rotation=45)
        plt.tight_layout()

    def price_and(self):
        """
        Scatter plots showing relationship between house buying price and various features.
        """

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(20, 10))
        ax1.scatter(self.df["built_year"], self.df["buy_price"], color="purple")
        ax1.set_title("Price and Year Built")
        ax1.set_xlabel("Year Built")
        ax1.set_ylabel("Buy Price")
        ax2.scatter(self.df["sq_mt_built"], self.df["buy_price"], color="purple")
        ax2.set_title("Price and Space")
        ax2.set_xlabel("Square Meters Built")
        ax2.set_ylabel("Buy Price")
        ax3.scatter(self.df["n_bathrooms"], self.df["buy_price"], color="purple")
        ax3.set_title("Price and number of Bathrooms")
        ax3.set_xlabel("Number of Bathrooms")
        ax3.set_ylabel("Buy Price")
        ax4.scatter(self.df["n_rooms"], self.df["buy_price"], color="purple")
        ax4.set_title("Price and number of Rooms")
        ax4.set_xlabel("Number of Rooms")
        ax4.set_ylabel("Buy Price")

        plt.tight_layout()
