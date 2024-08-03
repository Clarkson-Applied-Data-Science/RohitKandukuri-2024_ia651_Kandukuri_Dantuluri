# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# %%
df = pd.read_csv("spotify.csv")
df

# %%
df.info()

# %%
spotify_data_dummies = pd.get_dummies(df, columns=['track_genre'])

print(spotify_data_dummies.columns)

# %%
plt.figure(figsize=(25, 10))
sns.scatterplot(x='duration_ms', y='danceability', data=df)
plt.xlabel('duration_ms')
plt.ylabel('danceability')
plt.title('Scatter plot of Danceability by popularity')
plt.xticks(rotation=90)
plt.show()

# %%
unique_genres = df['track_genre'].unique()
unique_genres = df['track_genre'].drop_duplicates()
unique_genres = set(df['track_genre'])
unique_genres = list(unique_genres)

genre_counts = df['track_genre'].value_counts()
print(genre_counts)

# %%
sns.pairplot(df, kind='reg', diag_kind='kde')
plt.suptitle('Pairplot with Regression Lines', y=1.02)  
plt.show()

# %%


numeric_columns = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'], s=5, alpha=0.9)
plt.title('PCA: Principal Components 1 vs 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# %%
corr_matrix = df[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=0, vmax=1)
plt.title('Correlation Heatmap of Music Track Attributes')
plt.show()


# %%
plt.figure(figsize=(20, 10))
sns.boxplot(x='track_genre', y='danceability', data=df)
plt.xlabel('Track Genre')
plt.ylabel('Danceability')
plt.title('Box Plot of Danceability by Genre')
plt.xticks(rotation=90)
plt.show()


# %%
plt.figure(figsize=(20, 10))
sns.boxplot(x='popularity', y='danceability', data=df)
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.title('Box Plot of Danceability by popularity')
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(15, 10))
plt.scatter(df['energy'], df['loudness'], s=10, c='brown', alpha=0.8)
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.title('Scatter Plot of Energy vs Loudness')
plt.show()

# %%
features = ['duration_ms', 'danceability', 'energy', 'loudness', 'tempo', 
            'explicit', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'time_signature']
target = 'popularity'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print(feature_importance_df)


# %%
data = df.copy()
finalData = data.drop(['Unnamed: 0','track_id','album_name','track_name'],axis=1)

# %%
finalData['popularity'] = pd.qcut(finalData['popularity'], q=2, labels=[0,1])

# %%
df_dummies = pd.get_dummies(df, columns=['track_genre'])

df_dummies = df_dummies.drop(['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'], axis=1)

y = df_dummies.loc[:, 'popularity']
X = df_dummies.drop(['popularity'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                      'time_signature']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training Set Evaluation")
print(f"Mean Absolute Error: {train_mae}")
print(f"Mean Squared Error: {train_mse}")
print(f"R-squared Score: {train_r2}")

print("\nTesting Set Evaluation")
print(f"Mean Absolute Error: {test_mae}")
print(f"Mean Squared Error: {test_mse}")
print(f"R-squared Score: {test_r2}")

# %%
feature_importance = np.abs(linear_model.coef_)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df.head(10))

# %%
rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation R-squared Score: {best_score}")

# %%
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

svr_model = SVR()
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.1, 0.2, 0.5, 0.3],
}

grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation R-squared Score: {best_score}")

# %%
def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

n_train = X_train.shape[0]
k_train = X_train.shape[1]
train_adj_r2 = adjusted_r2_score(train_r2, n_train, k_train)

n_test = X_test.shape[0]
k_test = X_test.shape[1]
test_adj_r2 = adjusted_r2_score(test_r2, n_test, k_test)

print(f"Adjusted R-squared (Train): {train_adj_r2}")
print(f"Adjusted R-squared (Test): {test_adj_r2}")

# %%
rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation R-squared Score: {best_score}")

best_rf_model = grid_search.best_estimator_
y_pred_train_rf = best_rf_model.predict(X_train)
y_pred_test_rf = best_rf_model.predict(X_test)

train_mae_rf = mean_absolute_error(y_train, y_pred_train_rf)
train_mse_rf = mean_squared_error(y_train, y_pred_train_rf)
train_r2_rf = r2_score(y_train, y_pred_train_rf)

test_mae_rf = mean_absolute_error(y_test, y_pred_test_rf)
test_mse_rf = mean_squared_error(y_test, y_pred_test_rf)
test_r2_rf = r2_score(y_test, y_pred_test_rf)

print(f"Training Set Evaluation (RandomForest)")
print(f"Mean Absolute Error: {train_mae_rf}")
print(f"Mean Squared Error: {train_mse_rf}")
print(f"R-squared Score: {train_r2_rf}")

print("\nTesting Set Evaluation (RandomForest)")
print(f"Mean Absolute Error: {test_mae_rf}")
print(f"Mean Squared Error: {test_mse_rf}")
print(f"R-squared Score: {test_r2_rf}")


