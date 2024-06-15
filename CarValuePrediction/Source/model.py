import pandas as pd
import numpy as np

df = pd.read_csv('data/used_car_canada_clean.csv')
# print(df.head())

cols_to_drop = ['body_type', 'vehicle_type', 'drivetrain', 'transmission', 'fuel_type', 'engine_block']
df = df.drop(cols_to_drop, axis=1)

# print(df.head())

df_toyota_honda = df.loc[(df['make'] == 'honda') | (df['make'] == 'toyota')]
# print(df_toyota_honda.head())

# df_toyota_honda.to_csv('data/honda_toyota_ca.csv', index=False, header=True)

### Model ###
df = pd.read_csv('data/honda_toyota_ca.csv')
# print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df[['make', 'model']], test_size=0.2, shuffle=True, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

cat_index = [2,3,5]

cat_features_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_features_transformer, cat_index)
    ]
)


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
                                                                   OneHotEncoder())]),
                                                  [2, 3, 5])])),
                ('regressor', GradientBoostingRegressor(random_state=42))])
# print(model.score(X_test, y_test))

from joblib import dump

dump(model, 'model/model.joblib')

