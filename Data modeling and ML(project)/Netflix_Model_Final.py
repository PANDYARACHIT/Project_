#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:03:37 2024

@author: g.dinkneh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import ast

df = pd.read_csv('titles.csv')

df.columns

df.dropna(subset=['imdb_score'], inplace=True)
df.dropna(subset=['tmdb_popularity'], inplace=True)


# Simplify genres and production countries by extracting the first item directly
df['first_genre'] = df['genres'].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) and ast.literal_eval(x) else 'Unknown')
df['first_country'] = df['production_countries'].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) and ast.literal_eval(x) else 'Unknown')

# Filter for specific age certifications
df_filtered = df[df['age_certification'].isin(['TV-MA', 'R', 'PG-13', 'TV-14', 'PG'])]

# Create dummy variables for 'first_genre', 'first_country', and 'age_certification'
dummies = pd.get_dummies(df_filtered[['first_genre', 'first_country', 'age_certification']], drop_first=True)

##based on this, we'll refine the model and drop some predictors

columns_to_keep = dummies.columns[dummies.sum() >= 50]

# Filter your DataFrame to keep only the columns that meet the threshold
dummies_filtered = dummies[columns_to_keep]

# Now, dummies_filtered contains only the columns where the dummy variable is True in 50 or more rows.

# Join the dummies back to the original dataframe
df_final = df_filtered.join(dummies)

# Join the dummies back to the original dataframe
df_final_filtered = df_filtered.join(dummies_filtered)


'''adding how many columns there are to decide which columns to drop'''

import pandas as pd

# Calculate the sum of True values for each column
true_counts = dummies.sum(axis=0)

# Sort the results by count in descending order
true_counts_sorted = true_counts.sort_values(ascending=False)

# Print the column names and corresponding summations
for column_name, true_count in true_counts_sorted.items():
    print(f"Column: {column_name}, Sum of True Values: {true_count}")


# Now df_final contains original columns plus binary variables for the categories

# We Assume 'imdb_score' and 'tmdb_popularity' are our target variables
X = df_final_filtered[dummies_filtered.columns]  # Features are the dummy variables
y_imdb = df_final_filtered['imdb_score']  # Target variable for IMDb scores
y_tmdb = df_final_filtered['tmdb_popularity']  # Target variable for TMDB popularity


#Visualizing the Dummy Variables
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame with dummy variables is df
# Calculate the sum of True values for each column
true_counts = dummies_filtered.sum(axis=0)

# Sort the results by count in descending order
true_counts_sorted = true_counts.sort_values(ascending=False)

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
true_counts_sorted.plot(kind='bar', color='skyblue')
plt.xlabel('Dummy Variables')
plt.ylabel('Count of True Values')
plt.title('Count of True Values in Each Column')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(axis='y')  # Add gridlines for the y-axis
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


#Build the Linera Regression Model and Train it

from sklearn.model_selection import train_test_split

# Splitting the dataset for IMDb score prediction
X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb, test_size=0.2, random_state=42)

# Splitting the dataset for TMDB popularity prediction
X_train_tmdb, X_test_tmdb, y_train_tmdb, y_test_tmdb = train_test_split(X, y_tmdb, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression model for IMDb score
model_imdb = LinearRegression()
model_imdb.fit(X_train_imdb, y_train_imdb)

# Linear Regression model for TMDB popularity
model_tmdb = LinearRegression()
model_tmdb.fit(X_train_tmdb, y_train_tmdb)


#Evaluate the Model

# Predicting and evaluating for IMDb score
y_pred_imdb = model_imdb.predict(X_test_imdb)
mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
r2_imdb = r2_score(y_test_imdb, y_pred_imdb)

# Predicting and evaluating for TMDB popularity
y_pred_tmdb = model_tmdb.predict(X_test_tmdb)
mse_tmdb = mean_squared_error(y_test_tmdb, y_pred_tmdb)
r2_tmdb = r2_score(y_test_tmdb, y_pred_tmdb)

# Printing the evaluation results
print(f"IMDb Score Model - MSE: {mse_imdb}, R-squared: {r2_imdb}")
print(f"TMDB Popularity Model - MSE: {mse_tmdb}, R-squared: {r2_tmdb}")


#Visualizing the model for IMDB

coef_imdb = pd.Series(model_imdb.coef_, index=X.columns)

plt.figure(figsize=(10, 6))
coef_imdb.sort_values().plot(kind='barh')
plt.title('Coefficients in the IMDb Score Model')
plt.show()

##Refine the model


#Visualizing the model for tmdb
import matplotlib.pyplot as plt
import pandas as pd


# Extracting the model coefficients and feature names
tmdb_coef = model_tmdb.coef_
feature_names = X_train_tmdb.columns 

# Creating a DataFrame to hold the coefficients and feature names
coefficients_tmdb = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': tmdb_coef
})

# Sorting the coefficients for better visualization
coefficients_tmdb_sorted = coefficients_tmdb.sort_values(by='Coefficient', ascending=True)

# Plotting the coefficients
plt.figure(figsize=(10, 8))
plt.barh(coefficients_tmdb_sorted['Feature'], coefficients_tmdb_sorted['Coefficient'])
plt.title('Coefficients in the TMDB Popularity Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()  # Adjust the layout to make room for feature labels if necessary
plt.show()




'''Decision Tree: let's test additional models'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Splitting the dataset for IMDb score prediction
X_train, X_test, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb, test_size=0.2, random_state=42)

# Splitting the dataset for TMDB popularity prediction
X_train, X_test, y_train_tmdb, y_test_tmdb = train_test_split(X, y_tmdb, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor for IMDb score
dt_regressor_imdb = DecisionTreeRegressor(random_state=42)
dt_regressor_tmdb = DecisionTreeRegressor(random_state=42)

# Train the regressor with the training data
dt_regressor_imdb.fit(X_train, y_train_imdb)
dt_regressor_tmdb.fit(X_train, y_train_tmdb)

# Predict the IMDb scores
y_pred_imdb = dt_regressor_imdb.predict(X_test)
y_pred_tmdb = dt_regressor_tmdb.predict(X_test)

# Evaluate the model
mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
r2_imdb = r2_score(y_test_imdb, y_pred_imdb)

mse_tmdb = mean_squared_error(y_test_tmdb, y_pred_tmdb)
r2_tmdb = r2_score(y_test_tmdb, y_pred_tmdb)
print(f"Decision Tree Regressor - IMDb Score - MSE: {mse_imdb}, R-squared: {r2_imdb}")
print(f"Decision Tree Regressor - TMDb Popularity - MSE: {mse_tmdb}, R-squared: {r2_tmdb}")



'''Hyperparameter tuning using cross validation'''
#IMDB Score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Define the hyperparameter grid to search
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize a DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=42)

# Set up the GridSearchCV object
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# We'll use training data in X_train and y_train
# Perform the grid search
grid_search.fit(X_train, y_train_imdb)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate on the testing data
y_pred_imdb = best_model.predict(X_test)
mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
r2_imdb = r2_score(y_test_imdb, y_pred_imdb)

print(f"Best Parameters: {best_params}")
print(f"Test MSE: {mse_imdb}")
print(f"Test R-squared: {r2_imdb}")

#TMDB Popularity

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the hyperparameter grid to search
param_grid_tmdb = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize the DecisionTreeRegressor
dtree_tmdb = DecisionTreeRegressor(random_state=42)

# Set up the GridSearchCV object
grid_search_tmdb = GridSearchCV(estimator=dtree_tmdb, param_grid=param_grid_tmdb, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Perform the grid search on the training data
grid_search_tmdb.fit(X_train_tmdb, y_train_tmdb)

# Get the best parameters and best model from the grid search
best_params_tmdb = grid_search_tmdb.best_params_
best_model_tmdb = grid_search_tmdb.best_estimator_

# Predict on the test data using the best model
y_pred_tmdb = best_model_tmdb.predict(X_test_tmdb)

# Calculate the performance metrics
mse_tmdb = mean_squared_error(y_test_tmdb, y_pred_tmdb)
r2_tmdb = r2_score(y_test_tmdb, y_pred_tmdb)

# Print out the best parameters and performance metrics
print(f"Best Parameters for TMDB Popularity: {best_params_tmdb}")
print(f"Test MSE for TMDB Popularity: {mse_tmdb}")
print(f"Test R-squared for TMDB Popularity: {r2_tmdb}")
























