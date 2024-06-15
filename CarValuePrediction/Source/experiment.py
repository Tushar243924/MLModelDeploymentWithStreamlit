# Data source: https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada?select=ca-dealers-used.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/ca-dealers-used.csv', low_memory=False)
# print(df.head())
# print(df.isna().sum())
# print(df.shape)

# Drop columns
cols_to_drop = ['id', 'vin', 'trim', 'stock_no', 'street', 'zip', 'seller_name', 'city']
df = df.drop(cols_to_drop, axis=1)

# print(df.shape)
# print(df.dtypes)

# Drop row having NA values
df = df.dropna(axis=0)
# print(df.isna().sum())

# Change data type for colunns from object to string
df[['make', 'model', 'body_type', 'vehicle_type', 'drivetrain', 'transmission', 'fuel_type', 'engine_block', 'state']] = df[['make', 'model', 'body_type', 'vehicle_type', 'drivetrain', 'transmission', 'fuel_type', 'engine_block', 'state']].astype("string")

# print(df.dtypes)
# print(df.shape)
# print(df.head())
# print(df['make'].unique())

# Use lamda function to lower case, replace '-' with '_' and replace ' ' with '_' for values in 'make'
df['make'] = df['make'].apply(lambda x : x.lower().replace(' ', '_').replace('-', '_'))
# print(df['make'].unique())

# Use lamda function to lower case, replace '-' with '_' and replace ' ' with '_' for values in 'body_type'
df['body_type'] = df['body_type'].apply(lambda x : x.lower().replace(' ', '_').replace('-', '_'))
# print(df['body_type'].unique())

# print(df['vehicle_type'].unique())
# print(df['drivetrain'].unique())
# print(df['transmission'].unique())
# print(df['fuel_type'].unique())

# Update 'fuel_type' in category gasoline, biodiesel, hybrid and other
gasoline = {'gas', 
            'E85 / Unleaded',
       'Unleaded', 'Premium Unleaded',
       'Premium Unleaded; Unleaded', 
       'Unleaded; Unleaded / E85', 'Unleaded / E85',
       'E85 / Unleaded; Unleaded', 'Premium Unleaded / Unleaded',
       'E85 / Premium Unleaded; E85 / Unleaded', 
       'E85', 'E85 / Premium Unleaded', 
       'Compressed Natural Gas; Unleaded',
       'E85 / Unleaded; Unleaded / Unleaded',
       'Diesel / Premium Unleaded', 'E85 / Unleaded; Unleaded / E85',
       'Unleaded / Unleaded', 
       'Compressed Natural Gas / Unleaded', 'Diesel; Unleaded',
       'Diesel; E85 / Unleaded', 'E85 / Unleaded; Premium Unleaded',
       'Premium Unleaded; Premium Unleaded / E85', 'E85; E85 / Unleaded',
       'Unleaded / Premium Unleaded',
       'Premium Unleaded / E85',
       'M85 / Unleaded'
}

biodiesel = {
    'Biodiesel'
}

hybrid = {
       'Electric / Premium Unleaded', 
       'Electric / Unleaded',
       'Unleaded / Electric',
       'Electric / Hydrogen',
       'Electric / Premium Unleaded; Electric / Unleaded',
       'Electric / Premium Unleaded; Premium Unleaded',
       'Electric / E85'
}

other = {
    'Hydrogen', 
    'Premium Unleaded / Natural Gas',
    'Compressed Natural Gas / Lpg', 
    'Compressed Natural Gas', 'Propane',
    'Flex Fuel Vehicle',
}

def preprocess_fuel_type(x):
    if x in gasoline:
        return "gasoline"
    elif x in biodiesel:
        return "biodiesel"
    elif x in hybrid:
        return "hybrid"
    else:
        return "other"
    
df['fuel_type'] = df['fuel_type'].apply(preprocess_fuel_type)

# print(df['fuel_type'].unique())

# print(df['engine_size'].unique())
# print(df['engine_block'].unique())
# print(df['state'].unique())

unique_make_model = df.groupby(['make', 'model']).size().reset_index(name='count')
# print(unique_make_model.head())

non_unique_rows = unique_make_model[unique_make_model['count']>1]
# print(non_unique_rows.head())

df_filtered = df.merge(non_unique_rows, on=['make', 'model'], how='inner').drop(['count'], axis=1)
# print(df_filtered.head())

# print(df_filtered.shape)

# df_filtered.to_csv('data/used_car_canada_clean.csv', index=False, header=True)

### Modeling ###
df = pd.read_csv('data/used_car_canada_clean.csv')
# print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df[['make', 'model']], test_size=0.2, shuffle=True, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectPercentile, mutual_info_regression

cat_index = [3,4,5,6,7,10,11]

cat_features_transformer = Pipeline(
    steps=[
        ("encoder", OrdinalEncoder()),
        ("selector", SelectPercentile(mutual_info_regression, percentile=50))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_features_transformer, cat_index)
    ]
)

from sklearn.ensemble import GradientBoostingRegressor

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ]
)

model.fit(X_train, y_train)

Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('cat',
                                                  Pipeline(steps=[('encoder',
                                                                   OrdinalEncoder()),
                                                                  ('selector',
                                                                   SelectPercentile(percentile=50))]),
                                                  [3, 4, 5, 6, 7, 8, 10,
                                                   11])])),
                ('regressor', GradientBoostingRegressor(random_state=42))])

# Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('cat',
#                                                                   Pipeline(steps=[('encoder',
#                                                                                    OrdinalEncoder()),
#                                                                                    ('selector',
#                                                                                     SelectPercentile(percentile=50,
#                                                                                                      score_func =<function mutual_info_regression at 0x000001CEC8066050> ))]),
#                                                                                                      [3,4,5,6,7,8,10,11])])),
#                                                                                                      ('regressor', GradientBoostingRegressor(random_state=42))])

print(model.score(X_test, y_test))




