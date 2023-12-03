import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

# Load the dataset
dataset_path = path\to\your\file.csv
data = pd.read_csv(dataset_path)

# Split the dataset into features and target variable
X = data.drop(columns=['target_variable'])
y = data['target_variable']

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Define the number of features for Mutual Information feature selection
num_features_list = [50, 100, 150, 200]

best_weight = 0
best_mean_test_score = float('-inf')

for num_features in num_features_list:
    for model_name, model_class in [
        ('KNN', KNeighborsRegressor()), 
        ('Elastic Net Linear Regression', ElasticNet()),
        ('Linear Regression', LinearRegression()), 
        ('SVR Linear', SVR(kernel='linear')), 
        ('SVR RBF', SVR(kernel='rbf')),
        ('SVR Sigmoid', SVR(kernel='sigmoid')),
        ('Nu-SVR', NuSVR()),
        ('Decision Tree', DecisionTreeRegressor(random_state=1)), 
        ('Gradient Boosting', GradientBoostingRegressor(random_state=1)),
        ('XGBoost', XGBRegressor(random_state=1)),  
    ]:
        print(f"Number of Features: {num_features}")
        print(f"Model: {model_name}")

        # Additional hyperparameters for Elastic Net Linear Regression
        alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
        l1_ratios = [(0.99, 0.01), (0.5, 0.5)]
        
        # Exclude Random Forest and AdaBoost from the current iteration
        if model_name not in ['Random Forest', 'AdaBoost']:
            # Create a SelectKBest transformer with Mutual Information as the scoring function
            feature_selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
            
            model_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selector', feature_selector),
                ('model', model_class)
            ])
        
            model_param_grid = {}
            if model_name == 'KNN':
                model_param_grid = {
                    'model__n_neighbors': [5, 10, 15, 20],
                    'model__weights': ['uniform', 'distance']
                }
            elif model_name == 'Decision Tree':  
                model_param_grid = {
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif model_name.startswith('SVR') or model_name == 'Nu-SVR':
                if model_name == 'SVR Linear':
                    model_param_grid = {
                        'model__C': [0.1, 1, 10, 100]
                    }
                elif model_name == 'SVR RBF':
                    model_param_grid = {
                        'model__C': [0.1, 1, 10, 100],
                        'model__gamma': [0.1, 1, 10]
                    }
                elif model_name == 'SVR Sigmoid':
                    model_param_grid = {
                        'model__C': [0.1, 1, 10],
                        'model__gamma': [0.1, 1, 10],
                        'model__coef0': [-1, 0, 1]
                    }
                elif model_name == 'Nu-SVR':
                    model_param_grid = {
                        'model__C': [0.1, 1, 10, 100],
                        'model__nu': [0.1, 0.3, 0.5, 0.7, 0.9],
                        'model__kernel': ['linear', 'rbf', 'sigmoid']
                    }
        
            model_grid_search = GridSearchCV(model_pipeline, model_param_grid,
                                             scoring='neg_mean_squared_error', cv=5)
            model_grid_search.fit(X_train, y_train)
            
            best_model = model_grid_search.best_estimator_
            best_model_params = model_grid_search.best_params_
            
            model_test_score = best_model.score(X_test, y_test)
            
            y_pred = best_model.predict(X_test)
            r2_train = max(0, r2_score(y_train, best_model.predict(X_train)))
            r2_val = max(0, r2_score(y_test, y_pred))
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            w_new = ((r2_train + r2_val) / (mse + rmse + mae)) * (
                1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)) / (
                    1 + ((r2_train + r2_val) / (mse + rmse + mae)) * (
                    1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)))
    
            if r2_train >= 0 and r2_val >= 0 and model_grid_search.best_score_ > best_mean_test_score:
                best_mean_test_score = model_grid_search.best_score_
                best_weight = w_new
    
            print("Best model:")
            print(best_model)
            print("Best model parameters:")
            print(best_model_params)
            print("R2 Train Score:", r2_train)
            print("R2 Validation Score:", r2_val)
            print("Mean Absolute Error:", mae)
            print("Root Mean Squared Error:", rmse)
            print("Mean Squared Error:", mse)
            print("Weight (new):", w_new)
            print("=" * 50)
        
    # Random Forest
    print(f"Number of Features: {num_features}")
    print(f"Model: Random Forest")

    # Create the model
    model = RandomForestRegressor(random_state=1)

    # Create the parameter grid
    param_grid = {
        'model__n_estimators': [5, 10, 100],
        'model__max_depth': [3, 5, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 5, 10],
    }

    # Create a pipeline with StandardScaler and SelectKBest with Mutual Information
    feature_selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
    
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

    # Create GridSearchCV object
    model_grid_search = GridSearchCV(model_pipeline, param_grid,
                                 scoring='neg_mean_squared_error', cv=5)

    # Fit the model
    model_grid_search.fit(X_train, y_train)

    best_model = model_grid_search.best_estimator_
    best_model_params = model_grid_search.best_params_

    model_test_score = best_model.score(X_test, y_test)

    y_pred = best_model.predict(X_test)
    r2_train = max(0, r2_score(y_train, best_model.predict(X_train)))
    r2_val = max(0, r2_score(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    w_new = ((r2_train + r2_val) / (mse + rmse + mae)) * (
       1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)) / (
           1 + ((r2_train + r2_val) / (mse + rmse + mae)) * (
           1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)))

    if r2_train >= 0 and r2_val >= 0 and model_grid_search.best_score_ > best_mean_test_score:
        best_mean_test_score = model_grid_search.best_score_
        best_weight = w_new

    print("Best model:")
    print(best_model)
    print("Best model parameters:")
    print(best_model_params)
    print("R2 Train Score:", r2_train)
    print("R2 Validation Score:", r2_val)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("Mean Squared Error:", mse)
    print("Weight (new):", w_new)
    print("=" * 50)

    # AdaBoost
    print(f"Number of Features: {num_features}")
    print(f"Model: AdaBoost")

    # Create the model
    model = AdaBoostRegressor(random_state=1)

    # Create the parameter grid
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__loss': ['linear', 'square', 'exponential'],
        'model__random_state': [1],
    }

    # Create a pipeline with StandardScaler and SelectKBest with Mutual Information
    feature_selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
    
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

    # Create GridSearchCV object
    model_grid_search = GridSearchCV(model_pipeline, param_grid,
                                     scoring='neg_mean_squared_error', cv=5)

    # Fit the model
    model_grid_search.fit(X_train, y_train)

    best_model = model_grid_search.best_estimator_
    best_model_params = model_grid_search.best_params_

    model_test_score = best_model.score(X_test, y_test)

    y_pred = best_model.predict(X_test)
    r2_train = max(0, r2_score(y_train, best_model.predict(X_train)))
    r2_val = max(0, r2_score(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    w_new = ((r2_train + r2_val) / (mse + rmse + mae)) * (
        1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)) / (
            1 + ((r2_train + r2_val) / (mse + rmse + mae)) * (
            1 - abs(r2_train - r2_val)) / (1 + abs(r2_train - r2_val)))

    if r2_train >= 0 and r2_val >= 0 and model_grid_search.best_score_ > best_mean_test_score:
        best_mean_test_score = model_grid_search.best_score_
        best_weight = w_new

    print("Best model:")
    print(best_model)
    print("Best model parameters:")
    print(best_model_params)
    print("R2 Train Score:", r2_train)
    print("R2 Validation Score:", r2_val)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("Mean Squared Error:", mse)
    print("Weight (new):", w_new)
    print("=" * 50)

print("Best Weight:", best_weight)
print("Best Mean Test Score:", best_mean_test_score)
