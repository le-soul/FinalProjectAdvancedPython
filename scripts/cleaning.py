"""
Here is the cleaning class located for the cleaning of data
"""


class CleaningClass:
    """
    Class for viewing null values and duplicates, and deciding on deleting them
    """
    
    def __init__(self, df):
        self.df = df

    def display_null_and_duplicates_info(self):
        """
        Displays null and duplicate values
        """
        null_info = self.df.isnull().sum()
        print("Null Values Information:")
        print(null_info)
        duplicate_info = self.df.duplicated().sum()
        print("Duplicate Values Information:")
        print(duplicate_info)

    def clean_data(self):
        """
        Cleans the data to be able to conduct analysis
        """
        self.df = self.df.drop(columns=['Unnamed: 0'])
        self.df = self.df.sort_index(axis=0)
        self.df = self.df.drop(columns=['title','subtitle','sq_mt_useful','n_floors','sq_mt_allotment','latitude','longitude','raw_address',
                    'is_exact_address_hidden','street_name','street_number','portal','floor','is_floor_under','door','operation','rent_price',
                    'rent_price_by_area','is_rent_price_known','buy_price_by_area','is_buy_price_known','has_central_heating','has_individual_heating',
                    'are_pets_allowed','is_furnished','is_kitchen_equipped','has_garden','is_renewal_needed','energy_certificate','has_private_parking','has_public_parking',
                    'is_parking_included_in_price','parking_price','is_orientation_north','is_orientation_west','is_orientation_south','is_orientation_east'])
        collist=['has_ac','has_fitted_wardrobes','has_pool','has_terrace','has_balcony','has_storage_room','is_accessible','has_green_zones']
        for col in collist:
            self.df[col]=self.df[col].fillna(False)

        self.df=self.df[self.df['sq_mt_built'].notna()]
        self.df=self.df[self.df['n_bathrooms'].notna()]

        mask=(self.df['is_new_development'].isnull()) & (~self.df['built_year'].isnull())
        self.df['is_new_development'][mask]=False

        self.df['district_id'] = self.df['neighborhood_id'].copy()
        self.df.district_id = self.df.district_id.str.extract(r'(District \d+)')
        self.df.neighborhood_id = self.df.neighborhood_id.str.extract(r'(Neighborhood \d+)')
        self.df.district_id = self.df.district_id.str.extract(r'(\d+)')
        self.df.neighborhood_id = self.df.neighborhood_id.str.extract(r'(\d+)')
        self.df.drop(columns='neighborhood_id')
        self.df = self.df[self.df['built_year'] != 8170]
        self.df['has_parking'] = self.df['has_parking'].astype(int)
        self.df['has_ac'] = self.df['has_ac'].astype(int)
        self.df['district_id'] = self.df['district_id'].astype(int)

        return self.df


