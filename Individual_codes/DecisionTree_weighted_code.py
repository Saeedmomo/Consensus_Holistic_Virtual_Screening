from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)

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
model_name = 'Decision Tree'
model_class = DecisionTreeRegressor(random_state=1)

best_w_new_global = -float('inf')  # Initialize with negative infinity
best_params_global = None

for num_components in num_components_list:
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=num_components)),
        ('model', model_class)
    ])
    
    model_param_grid = {
        'model__max_depth': [None, 2, 5, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 5, 10]
    }
    
    model_grid_search = GridSearchCV(model_pipeline, model_param_grid, scoring='neg_mean_squared_error', cv=5)
    model_grid_search.fit(X_train, y_train)
    
    max_w_new_iteration = -float('inf')  # Initialize with negative infinity for each iteration
    
    print(f"Model: {model_name}, PCA Components: {num_components}")
    
    for i in range(len(model_grid_search.cv_results_['params'])):
        # Set the current hyperparameters
        model_pipeline.set_params(**model_grid_search.cv_results_['params'][i])
        
        # Fit the model on training data
        model_pipeline.fit(X_train, y_train)
        
        # Calculate predictions on training and validation data
        y_train_pred = model_pipeline.predict(X_train)
        y_val_pred = model_pipeline.predict(X_test)
        
        # Calculate r2_train and r2_val for this iteration
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_test, y_val_pred)
        
        # Calculate mse, rmse, and mae for this iteration
        mse = mean_squared_error(y_test, y_val_pred)
        rmse = mean_squared_error(y_test, y_val_pred, squared=False)
        mae = mean_absolute_error(y_test, y_val_pred)
        
        # Compute the formula for 'w_new'
        curr_w_new = ((r2_train + r2_val) / (mse + rmse + mae)) * (
                1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)) / (
                1 + ((r2_train + r2_val) / (mse + rmse + mae)) * (
                1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)))
        
        if curr_w_new > max_w_new_iteration:
            max_w_new_iteration = curr_w_new  # Update max_w_new_iteration within the current iteration
        
        print("Model:", model_name, "- PCA Components:", num_components,
              model_grid_search.cv_results_['params'][i],
              "r2_train:", r2_train, "r2_val:", r2_val, "MSE:", mse, "RMSE:", rmse, "MAE:", mae, "w_new:", curr_w_new)
    
    if max_w_new_iteration > best_w_new_global:
        best_w_new_global = max_w_new_iteration  # Update best_w_new_global across iterations
        best_params_global = model_grid_search.best_params_
    
    print("Highest w_new in this iteration: ", max_w_new_iteration)
    print("===================")

print("Best parameters overall: ", best_params_global)
print("Highest w_new overall: ", best_w_new_global)
