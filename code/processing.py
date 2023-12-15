import pandas as pd
import numpy as np
import chardet

cuisine_types = [
    "Afghan", "African", "American", "Armenian", "Asian", "Australian", "Austrian", "Bagels",
    "Bakery", "Bar", "Bar_Pub_Brewery", "Barbecue", "Basque", "Brazilian", "Breakfast-Brunch",
    "British", "Burgers", "Burmese", "Cafe-Coffee_Shop", "Cafeteria", "Cajun-Creole", "California",
    "Cambodian", "Canadian", "Caribbean", "Chilean", "Chinese", "Contemporary", "Continental-European",
    "Cuban", "Deli-Sandwiches", "Dessert-Ice_Cream", "Dim_Sum", "Diner", "Doughnuts", "Dutch-Belgian",
    "Eastern_European", "Eclectic", "Ethiopian", "Family", "Fast_Food", "Filipino", "Fine_Dining",
    "French", "Fusion", "Game", "German", "Greek", "Hawaiian", "Hot_Dogs", "Hungarian",
    "Indian-Pakistani", "Indigenous", "Indonesian", "International", "Irish", "Israeli", "Italian",
    "Jamaican", "Japanese", "Juice", "Korean", "Kosher", "Latin_American", "Lebanese", "Malaysian",
    "Mediterranean", "Mexican", "Middle_Eastern", "Mongolian", "Moroccan", "North_African",
    "Organic-Healthy", "Pacific_Northwest", "Pacific_Rim", "Persian", "Peruvian", "Pizzeria", "Polish",
    "Polynesian", "Portuguese", "Regional", "Romanian", "Russian-Ckrainian", "Scandinavian", "Seafood",
    "Soup", "Southeast_Asian", "Southern", "Southwestern", "Spanish", "Steaks", "Sushi", "Swiss", "Tapas",
    "Tea_House", "Tex-Mex", "Thai", "Tibetan", "Tunisian", "Turkish", "Vegetarian", "Vietnamese"
]

cuisine_dict = {cuisine: index for index, cuisine in enumerate(cuisine_types)}

payment_dict = {
    'cash': 0,
    'gift_certificates': 0,
    'bank_debit_cards': 1,
    'VISA': 1,
    'Visa': 1,
    'MasterCard-Eurocard': 1,
    'American_Express': 1,
    'Discover': 1,
    'Diners_Club': 1
}


def calculate_average_hours(rows):

    total_hours = 0
    unique_days = set()

    for index, row in rows.iterrows():

        time_ranges = row['hours'].strip(';').split(';')
        days_open = row['days'].strip(';').split(';')

        for time_range in time_ranges:
            if '-' not in time_range:
                continue
            open_time, close_time = map(str.strip, time_range.split('-'))

            # Calculate the number of hours the restaurant is open
            hours_open = pd.to_datetime(close_time) - pd.to_datetime(open_time)
            total_hours += hours_open.seconds / 3600

        # Extract the days the restaurant is open
        unique_days.update(days_open)

    # Calculate the average number of hours and number of unique days
    num_unique_days = len(unique_days)
    average_hours = total_hours / num_unique_days
    return average_hours, num_unique_days


def clean_hours(df):

    grouped = df.groupby('placeID')

    restaurant_ids = df['placeID'].unique()
    average_hours_open, num_days_open = {}, {}
    for id in restaurant_ids:
        rows = grouped.get_group(id)
        average_hours, num_days = calculate_average_hours(rows)
        average_hours_open[id] = average_hours
        num_days_open[id] = num_days

    df['averageHoursOpen'] = df['placeID'].map(average_hours_open)
    df['numDaysOpen'] = df['placeID'].map(num_days_open)
    df = df.drop_duplicates(subset=['placeID'], keep='first')
    df = df.drop(columns={'days', 'hours'})
    return df


def merge_dfs():

    ratings = pd.read_csv("data/rating_final.csv")
    parking = pd.read_csv("data/chefmozparking.csv")
    hours = pd.read_csv("data/chefmozhours4.csv")
    restaurant_cuisine = pd.read_csv("data/chefmozcuisine.csv")
    user_cuisine_preferences = pd.read_csv("data/usercuisine.csv")
    restaurant_payments = pd.read_csv("data/chefmozaccepts.csv")
    with open('data/geoplaces2.csv', 'rb') as f:
        result = chardet.detect(f.read())

    restaurant_info = pd.read_csv(
        "data/geoplaces2.csv", encoding=result['encoding'])
    user_payment_preferences = pd.read_csv("data/userpayment.csv")
    user_profile = pd.read_csv("data/userprofile.csv")

    hours = clean_hours(hours)

    user_cuisine_preferences = user_cuisine_preferences.rename(columns={'Rcuisine' : 'Ccuisine'})

    restaurant_dfs = [parking, hours, restaurant_cuisine,
                      restaurant_payments, restaurant_info]
    user_dfs = [user_cuisine_preferences,
                user_payment_preferences, user_profile]

    for i, df in enumerate(restaurant_dfs):
        df.drop_duplicates(subset=['placeID'], keep='first', inplace=True)

    for i, df in enumerate(user_dfs):
        df.drop_duplicates(subset=['userID'], keep='first', inplace=True)

    ratings.set_index(['userID', 'placeID'], inplace=True)
    parking.set_index('placeID', inplace=True)
    restaurant_cuisine.set_index('placeID', inplace=True)
    restaurant_payments.set_index('placeID', inplace=True)
    restaurant_info.set_index('placeID', inplace=True)
    hours.set_index('placeID', inplace=True)
    user_cuisine_preferences.set_index('userID', inplace=True)
    user_payment_preferences.set_index('userID', inplace=True)
    user_profile.set_index('userID', inplace=True)

    merged_restaurant_dfs = pd.concat(restaurant_dfs, axis=1)
    merged_user_dfs = pd.concat(user_dfs, axis=1)
    return merged_restaurant_dfs, merged_user_dfs, ratings


def clean_user_df(merged_user_dfs):

    merged_user_dfs = merged_user_dfs.drop(columns=['latitude', 'longitude'])
    merged_user_dfs = merged_user_dfs.rename(
        columns={'hijos': 'Cchildren',
                 'smoker': 'Csmoker',
                 'drink_level': 'Cdrink',
                 'dress_preference': 'Cdress',
                 'ambience': 'Cambience',
                 'transport': 'Ctransport',
                 'marital_status': 'Cmaritalstatus',
                 'interest': 'Cinterest',
                 'personality': 'Cpersonality',
                 'religion': 'Creligion',
                 'activity': 'Cactivity',
                 'color': 'Ccolor',
                 'budget': 'Cbudget',
                 'Upayment': 'Cpayment'
                 })
    merged_user_dfs = merged_user_dfs.replace('?', np.nan)

    relabels = {
        'Cpayment': payment_dict,
        'Ccuisine': cuisine_dict,
        'Csmoker': {
            'false': 0,
            'true': 1
        },
        'Cdrink': {
            'abstemious': 0,
            'social drinker': 1,
            'casual drinker': 2
        },
        'Cdress': {
            'no preference': 0,
            'informal': 1,
            'formal': 2,
            'elegant': 3
        },
        'Cambience': {
            'solitary': 0,
            'family': 1,
            'friends': 2
        },
        'Ctransport': {
            'on foot': 0,
            'public': 1,
            'car owner': 2
        },
        'Cmaritalstatus': {
            'single': 0,
            'married': 1,
            'widow': 2
        },
        'Cchildren': {
            'dependent': 0,
            'independent': 1,
            'kids': 2
        },
        'Cinterest': {
            'none': 0,
            'retro': 1,
            'technology': 2,
            'eco-friendly': 3,
            'variety': 4
        },
        'Cpersonality': {
            'conformist': 0,
            'thrifty-protector': 1,
            'hard-worker': 2,
            'hunter-ostentatious': 3
        },
        'Creligion': {
            'none': 0,
            'Catholic': 1,
            'Christian': 2,
            'Mormon': 3,
            'Jewish': 4,
        },
        'Cactivity': {
            'unemployed': 0,
            'student': 1,
            'working-class': 2,
            'professional': 3
        },
        'Ccolor': {
            'black': 0,
            'red': 1,
            'blue': 2,
            'green': 3,
            'purple': 4,
            'orange': 5,
            'yellow': 6,
            'white': 7
        },
        'Cbudget': {
            'low': 0,
            'medium': 1,
            'high': 2
        }
    }

    merged_user_dfs = merged_user_dfs.replace(relabels)
    return merged_user_dfs


def clean_restaurant_df(df):

    df = df.drop(columns=['latitude', 'longitude', 'the_geom_meter', 'city',
                          'state', 'country', 'fax', 'zip', 'name', 'address', 'url'])

    df = df.rename(columns={
        'parking_lot': 'Rparking',
        'alcohol': 'Ralcohol',
        'smoking_area': 'Rsmoking',
        'dress_code': 'Rdress',
        'accessibility': 'Raccessibility',
        'price': 'Rprice',
        'franchise': 'Rfranchise',
        'area': 'Rarea',
        'other_services': 'Rotherservices'
    })

    relabels = {
        'Rparking': {  # no parking vs free parking vs valet
            'none': 0,
            'free': 1,
            'street': 1,
            'public': 1,
            'yes': 1,
            'valet parking': 2,
            'validated parking': 2,
            'fee': 2
        },
        'Rcuisine': cuisine_dict,
        'Rpayment': payment_dict,
        'Ralcohol': {
            'No_Alcohol_Served': 0,
            'Wine-Beer': 1,
            'Full_Bar': 2
        },
        'Rsmoking': {  # permitted vs not permitted
            'none': 0,
            'not permitted': 0,
            'only at bar': 1,
            'section': 1,
            'permitted': 1
        },
        'Rdress': {
            'informal': 0,
            'casual': 1,
            'formal': 2
        },
        'Raccessibility': {
            'no_accessibility': 0,
            'partially': 1,
            'completely': 2
        },
        'Rprice': {
            'low': 0,
            'medium': 1,
            'high': 2
        },
        'Rambience': {
            'quiet': 0,
            'familiar': 1
        },
        'Rfranchise': {
            'f': 0,
            't': 1
        },
        'Rarea': {
            'closed': 0,
            'open': 1
        },
        'Rotherservices': {
            'none': 0,
            'internet': 1,
            'Internet': 1,
            'variety': 2
        }
    }

    df = df.replace(relabels)

    unique_values = {}
    for col in df.columns:
        unique_values[col] = df[col].unique()

    return df


def get_clean_data():
    merged_restaurant_dfs, merged_user_dfs, final_ratings = merge_dfs()
    cleaned_restaurant_df = clean_restaurant_df(merged_restaurant_dfs)
    cleaned_user_df = clean_user_df(merged_user_dfs)
    final_ratings = final_ratings.reset_index()
    final_ratings = final_ratings.set_index('userID')
    merged_df = pd.merge(final_ratings, cleaned_user_df, on='userID')

    final_merged_df = pd.merge(
        merged_df, cleaned_restaurant_df, left_on='placeID', right_index=True, how='left')

    final_merged_df = final_merged_df.reset_index()

    return final_merged_df


if __name__ == "__main__":
    df = get_clean_data()
