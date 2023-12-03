# Consensus_Holistic_Virtual_Screening
New_Weight-Multi-Model-Search
This code performs a grid search to find the best model for predicting a continuous target variable and diverse machine learning models.

This repository contains Python code for model evaluation in regression tasks. It provides two approaches: one using Principal Component Analysis (PCA) for dimensionality reduction and another using Mutual Information for feature selection. The code explores various regression models and helps you select the best-performing model for your dataset.

# Usage
# PCA-Based Feature Selection
1. Load the Dataset: Replace dataset_path with the path to your dataset CSV file.

2. Data Splitting: Adjust the test size and random state in train_test_split to divide your data into training and testing sets.

3. Model Selection: Customize the list of regression models in the loop according to your needs.

4. Hyperparameter Tuning: Configure hyperparameters and their search ranges for each model.

5. Run the Code: Execute the code to perform PCA-based evaluation of the listed models.

# Mutual Information Feature Selection
1. Load the Dataset: Replace dataset_path with the path to your dataset CSV file.

2. Data Splitting: Adjust the test size and random state in train_test_split to divide your data into training and testing sets.

3. Model Selection: Customize the list of regression models in the loop according to your needs.

4. Feature Selection: Specify the desired numbers of features (e.g., 50, 100, 150, 200) in the num_features_list variable.

5. Run the Code: Execute the code to perform Mutual Information-based feature selection and evaluate the listed models for different numbers of features.

# Models Explored
The code includes the following regression models for evaluation:

* K-Nearest Neighbors (KNN)
* Elastic Net Linear Regression
* Linear Regression
* Support Vector Regression (SVR) with different kernels (Linear, RBF, Sigmoid)
* Nu-Support Vector Regression (Nu-SVR) with different kernels (Linear, RBF, Sigmoid, poly)
* Decision Tree
* Gradient Boosting
* XGBoost
* Random Forest
* AdaBoost
  
# Evaluation Metrics
The code calculates various regression evaluation metrics for each model, including:

* R-squared (R2) score on the training and validation sets
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Squared Error (MSE) All the above metrics are evaluated using the mathmatical formula that calculates weight for each model
* Weight (new) based on custom calculations A greater weight value indicates a robust model performance characterized by higher R2-training and R2-validation scores, diminished MAE, RMSE, and/or MSE values, and a reduced disparity between R2-train and R2-validation. In contrast, a lower value of w_new signifies less robust model performance, marked by lower R2-training and R2-validation scores, increased MAE, RMSE, and/or MSE values, and an expanded disparity between R2-train and R2-validation. The applicability domain of w_new extends to cases where both R2-training and R2-validation fall within the range of 0 to 1. Consequently, the code has been designed to exclude results in which R2-training and R2-validation values are outside this specified range.
  
# Fine-Tuning with Individual Weighted Models
In our research, we have incorporated individual weighted models as an integral part of our analysis. These models provide a deeper exploration into the performance of the models that contribute the highest weight values to the combined model. This approach allows us to fine-tune the search process, focusing on identifying the best parameters that align with the most suitable evaluation metrics. By dissecting the contributions of individual models, we gain valuable insights into their strengths and weaknesses, ultimately enhancing our ability to optimize model performance.

# Important Note on Weight Calculation (w_new)
The code calculates a custom weight (w_new) to assess model performance. This weight considers R2 scores, error metrics, and the difference between training and validation R2 scores. A higher w_new indicates better model performance. It is important to consider this weight when selecting the best-performing model.

Feel free to adapt and extend this code for your specific regression tasks and datasets.

For any questions or issues, please reach out to [Said Moshawih, saeedmomo@hotmail.com].
