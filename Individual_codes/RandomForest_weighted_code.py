import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(1)
random.seed(1)

# Load the dataset
dataset_path = path\to\your\file.csv
data = pd.read_csv(dataset_path)

# Split the dataset into features and target variable
X = data.drop(columns=['target_variable'])
y = data['target_variable']

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the number of PCA components
num_components_list = [5, 10, 12, 15, 17, 19, 20]

# Define model parameters
model_name = 'Random Forest'
model_class = RandomForestRegressor(random_state=1)

# Create separate PCA pipelines for each number of components
for num_components in num_components_list:
    print("\n-------------------\n")  # Separator for better readability
    print(f"Number of PCA components: {num_components}")
    print(f"Model: {model_name}")

    highest_w_new = 0.0  # Initialize the highest w_new value
    
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=num_components)),
        ('model', model_class)
    ])
    
    # Define hyperparameters
    model_param_grid = {
        'model__n_estimators': [5, 10, 100],
        'model__max_depth': [3, 5, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 5, 10],
    }
    
    model_grid_search = GridSearchCV(model_pipeline, model_param_grid,
                                     scoring='neg_mean_squared_error', cv=5)
    
    # Fit the model on training data
    model_grid_search.fit(X_train, y_train)
    
    
    # Iterate over each set of parameters
    for param in model_grid_search.cv_results_['params']:
        model_pipeline.set_params(**param)
        model_pipeline.fit(X_train, y_train)
        
        y_pred_train = model_pipeline.predict(X_train)
        y_pred_test = model_pipeline.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_test, y_pred_test)

        # Check if both r2_train and r2_val are between 0 and 1
        if 0 <= r2_train <= 1 and 0 <= r2_val <= 1:
           mse = mean_squared_error(y_test, y_pred_test)
           rmse = mean_squared_error(y_test, y_pred_test, squared=False)
           mae = mean_absolute_error(y_test, y_pred_test)
        
           w_new = ((r2_train + r2_val) / (mse + rmse + mae)) * (
                    1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)) / (
                    1 + ((r2_train + r2_val) / (mse + rmse + mae)) * (
                    1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val))) 
        
           # Update highest_w_new if a higher value is found
           if w_new > highest_w_new:
                highest_w_new = w_new

        print("Model:", model_name, "- PCA Components:", num_components)
        print("Hyperparameters:", param)
        print(f"R2 Score on Training Set: {r2_train}")
        print(f"R2 Score on Test Set: {r2_val}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"New Weight: {w_new}")
        
        if w_new > highest_w_new:
            highest_w_new = w_new
    
    print(f"Highest w_new in this iteration: {highest_w_new}")
    print("===================")
