import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import sklearn
from keras.src.datasets import imdb
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import missingno as msno
import datetime as dt
import plotly.express as px

data = pd.read_csv("listings.csv", sep=",")

print(data.head())
# first check of duplicated
# data after removing duplicates

# Removing identical duplicates
data.drop_duplicates()
print("data after removing")
print((data.drop_duplicates()).sort_values('id'))

# Display the entire DataFrame
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

print(data.head())

# investigate the data type of the variables
print(data.dtypes)
print(data.info())
nan_rows = data[data.isnull().any(axis=1)]
print(nan_rows)

print(data.describe())

# rating as histogram
sns.histplot(data['review_scores_rating'], bins=20)
pyplot.title('Distribution of listing ratings')
pyplot.show()

type_of_room = data['room_type'].unique()
print(type_of_room)

count_type_of_room = data['room_type'].value_counts()
print(count_type_of_room)

# remove the $ symbol from price variable to deal with it and change type to float
data["price"] = data['price'].str.strip("$")
data["price"] = data['price'].astype(float)
print(data.dtypes)

mean_of_price = data["price"].mean()
print("mean of the price: ", mean_of_price)

sns.displot(data['price'], bins=50)
# pyplot.show()

# change the date variables  to date type
date_variables = data[['listing_added', 'last_review']].head()
print(date_variables)
data['listing_added'] = pd.to_datetime(data['listing_added'], format='%Y-%m-%d')
data['last_review'] = pd.to_datetime(data['last_review'], format='%Y-%m-%d')
print(data.dtypes)

# limit the type of rooms
data['room_type'] = data['room_type'].str.lower()
data['room_type'] = data['room_type'].str.strip()

mappings = {'private room': 'Private Room',
            'private': 'Private Room',
            'entire home/apt': 'Entire place',
            'shared room': 'Shared room',
            'home': 'Entire place'}
data['room_type'] = data['room_type'].replace(mappings)
data['room_type'].unique()

# dummy variable
# Convert categorical variable into dummy/indicator variables.
dummy_variables = pd.get_dummies(data['room_type'], prefix='room_type')

# Concatenate the dummy columns with the original DataFrame
data = pd.concat([data, dummy_variables], axis=1)
print(data['room_type'].unique())

# add words count of description into data set
data['words_count'] = data['name'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# split the neighbourhood column
borough_neighbourhood = data['neighbourhood_full'].str.split(",", expand=True)
borough_neighbourhood.head()
data['city'] = borough_neighbourhood[0]
data['town'] = borough_neighbourhood[1]
# add to the data set
data['city'] = data['city'].str.strip()
data['town'] = data['town'].str.strip()
print(data[["neighbourhood_full", "city", "town"]].head())
data.drop('neighbourhood_full', axis=1, inplace=True)
print(data[["city", "town"]].head())

rate_over_five = data[data['rating'] > 5.0]
print("Rate over five: ", rate_over_five["rating"])

sns.displot(data[data['rating'] > 5.0], bins=20)
# pyplot.show()

# delete the rows
data.drop(data[data['rating'] > 5.0].index, inplace=True)

sns.displot(data[data['rating'] > 5.0], bins=20)
# pyplot.show()

# dealing with missing data
msno.bar(data)
# pyplot.show()

# fill missing data with null
data = data.fillna({'reviews_per_month': 0,
                    'number_of_stays': 0,
                    '5_stars': 0})
# dealing with missing data
msno.bar(data)
# pyplot.show()

# add column to distinguish rated or not
is_rated = np.where(data['rating'].isna() == True, 0, 1)
data['is_rated'] = is_rated

print(data.head())

lines_no_price = data[data['price'].isna()].count()
print(lines_no_price)

print(data[~data['price'].isna()].count())

sns.boxplot(x='room_type', y='price', data=data)
pyplot.ylim(0, 500)
pyplot.xlabel('Room Type')
pyplot.ylabel('Price')
# pyplot.show()

fig = px.box(x=data['room_type'], y=data['price'])
# fig.show()

mean_of_type_room = data.groupby('room_type').median()['price']
print(mean_of_type_room)

# assumption to set the price of the missing price data as of the mean
data.loc[(data['price'].isna()) & (data['room_type'] == 'Entire place'), 'price'] = 163.0
data.loc[(data['price'].isna()) & (data['room_type'] == 'Private Room'), 'price'] = 70.0
data.loc[(data['price'].isna()) & (data['room_type'] == 'Shared Room'), 'price'] = 50.0

var = data[data['price'].isna()]

print(data.isna().sum())

inconsistent_dates = data[data['listing_added'].dt.date > data['last_review'].dt.date]
print(inconsistent_dates['listing_added'], inconsistent_dates['last_review'])

# find the duplicate data
# Return boolean Series denoting duplicate rows
duplicates = data.duplicated(subset='listing_id', keep=False)
# print the duplicates
print("duplicated rows")
print(data[duplicates].sort_values('listing_id'))

# data after removing duplicates
data.drop_duplicates()
print("data after removing")
print((data.drop_duplicates()).sort_values('listing_id'))

# save the file in the directory
data.to_csv('processed_data.csv', index=False)
data.to_csv('/Users/alaa/Documents/code/R/processed_data.csv', index=False)
