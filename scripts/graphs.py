"""
Class made for the view of data
"""
import matplotlib.pyplot as plt
import seaborn as sns

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
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)

    def price_skewness(self):
        """
        Function to see the skewness in the buying price of houses in Madrid
        """

        sns.distplot(self.df['buy_price'])
        print("Skewness: %f" % self.df['buy_price'].skew())

    def most_exp_districts(self):
        """
        Function to see the most expensive districts in Madrid in average plotted
        """

        top_21_expensive_districts = self.df.groupby('district_id')['buy_price'].mean().nlargest(21)

        plt.figure(figsize=(10, 6))
        top_21_expensive_districts.plot(kind='bar', color='skyblue')
        plt.title('Top 21 Expensive Districts by Average Buy Price')
        plt.xlabel('District ID')
        plt.ylabel('Average Buy Price')
        plt.xticks(rotation=45)
        plt.tight_layout()

    def most_rooms_districts(self):
        """
        Function to see where bigger houses are most commonly located plotted
        """

        average_rooms_per_district = self.df.groupby('district_id')['n_rooms'].mean().nlargest(21)

        plt.figure(figsize=(10, 6))
        average_rooms_per_district.plot(kind='bar', color='lightgreen')
        plt.title('Average Number of Rooms by District')
        plt.xlabel('District ID')
        plt.ylabel('Average Number of Rooms')
        plt.xticks(rotation=45)
        plt.tight_layout()

    def most_bathrooms_districts(self):
        """
        Function see the number of bathrooms per district in average plotted
        """

        average_rooms_per_district = df.groupby('district_id')['n_bathrooms'].mean().nlargest(21)

        plt.figure(figsize=(10, 6))
        average_rooms_per_district.plot(kind='bar', color='#ffcccb')
        plt.title('Average Number of Bathrooms by District')
        plt.xlabel('District ID')
        plt.ylabel('Average Number of Bathrooms')
        plt.xticks(rotation=45)
        plt.tight_layout()

    def price_and(self):
        """
        Scatter plots showing relationship between house buying price and various features.
        """

        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(20, 4))
        ax1.scatter(self.df['built_year'], self.df['buy_price'])
        ax1.set_title('Price and Year Built')
        ax2.scatter(self.df['sq_mt_built'], self.df['buy_price'])
        ax2.set_title('Price and Space')
        ax3.scatter(self.df['n_bathrooms'], self.df['buy_price'])
        ax3.set_title('Price and number of Bathrooms')
        ax4.scatter(self.df['n_rooms'], self.df['buy_price'])
        ax4.set_title('Price and number of Rooms')

    