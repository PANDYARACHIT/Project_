#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:37:15 2024

@author: tanishababbar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast  # For safely evaluating strings containing Python literals

# Load your dataset
netflix_data = pd.read_csv('/Users/tanishababbar/Desktop/data Mining /netflix_cleaned_final .csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# 1. Bar Chart of Content by Type
plt.figure(figsize=(10, 6))
sns.barplot(x=netflix_data['type'].value_counts().index, y=netflix_data['type'].value_counts().values, palette="coolwarm")
plt.title('Number of Titles by Content Type', fontsize=15)
plt.xlabel('Content Type', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 2. Histogram of Release Years
plt.figure(figsize=(12, 6))
sns.histplot(netflix_data['release_year'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of Release Years', fontsize=15)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.show()

# 3. Box Plot of Runtimes
plt.figure(figsize=(10, 6))
sns.boxplot(x='type', y='runtime', data=netflix_data, palette="pastel")
plt.title('Distribution of Runtime by Content Type', fontsize=15)
plt.xlabel('Content Type', fontsize=12)
plt.ylabel('Runtime (minutes)', fontsize=12)
plt.show()

# 4. Bar Chart of Top Genres
genres_list = [genre for sublist in netflix_data['genres'].apply(ast.literal_eval) for genre in sublist]
genre_counts = Counter(genres_list)
genre_counts_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Genre', data=genre_counts_df, palette="viridis")
plt.title('Top 10 Genres on Netflix', fontsize=15)
plt.xlabel('Number of Titles', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.show()

# 5. Heatmap of Correlations
correlation_matrix = netflix_data[['runtime', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=2, square=True)
plt.title('Correlation Heatmap of Numerical Variables', fontsize=15)
plt.show()

# 6. Scatter Plot of IMDb Score vs. TMDb Popularity
plt.figure(figsize=(12, 8))
sns.scatterplot(x='imdb_score', y='tmdb_popularity', data=netflix_data, alpha=0.6, edgecolor=None, color='purple')
plt.title('IMDb Score vs. TMDb Popularity', fontsize=15)
plt.xlabel('IMDb Score', fontsize=12)
plt.ylabel('TMDb Popularity', fontsize=12)
plt.show()

# 7. Bar Chart of Content by Age Certification
plt.figure(figsize=(12, 6))
sns.barplot(x=netflix_data['age_certification'].value_counts().index, y=netflix_data['age_certification'].value_counts().values, palette="Set2")
plt.title('Number of Titles by Age Certification', fontsize=15)
plt.xlabel('Age Certification', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 8. Line Chart of Average Scores over Years
average_scores_by_year = netflix_data.groupby('release_year').agg({'imdb_score': 'mean', 'tmdb_score': 'mean'}).reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x='release_year', y='imdb_score', data=average_scores_by_year, marker='o', color='orange', label='IMDb Score')
sns.lineplot(x='release_year', y='tmdb_score', data=average_scores_by_year, marker='o', color='blue', label='TMDb Score')
plt.title('Average IMDb and TMDb Scores Over Years', fontsize=15)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.legend()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import ast



# Apply filters
netflix_data_filtered = netflix_data[netflix_data['genres'].apply(lambda x: len(ast.literal_eval(x)) <= 2)]
netflix_data_filtered = netflix_data_filtered[netflix_data_filtered['age_certification'].isin(['TV-MA', 'R', 'PG-13', 'TV-14', 'PG'])]

# Data Preparation
X_filtered = netflix_data_filtered[['release_year', 'runtime', 'imdb_score', 'age_certification']]
y_filtered = netflix_data_filtered['tmdb_popularity']

# Preprocessing
numerical_features = ['release_year', 'runtime', 'imdb_score']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['age_certification']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])

# Modeling
model_filtered = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

# Split the data
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Train the model
model_filtered.fit(X_train_filtered, y_train_filtered)

# Predict and evaluate
y_pred_filtered = model_filtered.predict(X_test_filtered)
rmse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered, squared=False)
r2_filtered = r2_score(y_test_filtered, y_pred_filtered)

print(f'Filtered RMSE: {rmse_filtered}, Filtered R^2: {r2_filtered}')



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import ast

# Load the dataset
netflix_data = pd.read_csv('path_to_your_file/netflix_cleaned_final.csv')

# Apply filters
netflix_data_filtered = netflix_data[netflix_data['genres'].apply(lambda x: len(ast.literal_eval(x)) <= 2)]
netflix_data_filtered = netflix_data_filtered[netflix_data_filtered['age_certification'].isin(['TV-MA', 'R', 'PG-13', 'TV-14', 'PG'])]

# Data Preparation
X_filtered = netflix_data_filtered[['release_year', 'runtime', 'imdb_score', 'age_certification']]
y_filtered = netflix_data_filtered['tmdb_popularity']

# Preprocessing
numerical_features = ['release_year', 'runtime', 'imdb_score']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['age_certification']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])

# Modeling
model_filtered = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

# Split the data
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Train the model
model_filtered.fit(X_train_filtered, y_train_filtered)

# Predict and evaluate
y_pred_filtered = model_filtered.predict(X_test_filtered)
rmse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered, squared=False)
r2_filtered = r2_score(y_test_filtered, y_pred_filtered)

print(f'Filtered RMSE: {rmse_filtered}, Filtered R^2: {r2_filtered}')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import ast



# Apply filters
netflix_data_filtered = netflix_data[netflix_data['genres'].apply(lambda x: len(ast.literal_eval(x)) <= 2)]
netflix_data_filtered = netflix_data_filtered[netflix_data_filtered['age_certification'].isin(['TV-MA', 'R', 'PG-13', 'TV-14', 'PG'])]

# Data Preparation
X_filtered = netflix_data_filtered[['release_year', 'runtime', 'imdb_score', 'age_certification']]
y_filtered = netflix_data_filtered['tmdb_popularity']

# Preprocessing
numerical_features = ['release_year', 'runtime', 'imdb_score']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['age_certification']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])

# Modeling
model_filtered = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

# Split the data
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Train the model
model_filtered.fit(X_train_filtered, y_train_filtered)

# Predict and evaluate
y_pred_filtered = model_filtered.predict(X_test_filtered)
rmse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered, squared=False)
r2_filtered = r2_score(y_test_filtered, y_pred_filtered)

print(f'Filtered RMSE: {rmse_filtered}, Filtered R^2: {r2_filtered}')
