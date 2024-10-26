Multi-Task Learning Model for House Price Prediction and Classification
This project constructs a multi-task learning model capable of handling house price prediction (regression) and house classification (classification). It leverages PyTorch Lightning for efficient data loading, model building, logging, early stopping, and training.

Table of Contents
Project Overview
Exploratory Data Analysis
Data Preprocessing
Model Architecture and Training
Results
Hyperparameter Tuning with Optuna
Final Model and Evaluation
Project Overview
This project utilizes the House Prices - Advanced Regression Techniques Dataset, which includes various features such as location, neighborhood, and house characteristics. The multi-task model performs:

Regression: Predicts house prices.
Classification: Categorizes houses based on engineered features, including 'House Style' and 'Bldg Type'.
Exploratory Data Analysis
Key EDA insights:

Sale Price Distribution: Right-skewed with most homes at lower prices.
Relationships:
GrLivArea vs. SalePrice: Larger homes tend to have higher prices.
Sale Price by Neighborhood: Northridge, Northridge Heights, and Stonebrook neighborhoods show higher average prices.
Visualizations:
Box plots for sale prices by year and quality.
Correlation Heatmap: Shows strong positive correlation of GrLivArea, TotalBsmtSF, and OverallQual with SalePrice.
Pair Plot: Highlights relationships between features.
Data Preprocessing
Classification Target Feature Creation: House Category feature is engineered from House Style, BldgType, and age of the house (calculated using Year Built or Year Remod/Add).
Data Cleaning:
Handling missing values.
Converting relevant columns to categorical types.
Splitting and Transformation:
Data split into training, validation, and test sets.
Scaling: Sale prices standardized with StandardScaler.
Encoding: House Category one-hot encoded for classification.
Model Architecture and Training
Architectures Tested
Model 1: Multi-Task Neural Network
Layers: Shared layers, ReLU activation, dropout, task-specific heads for regression and classification.
Evaluation:
MSE: 0.252
Accuracy: 0.81
ROC AUC: 0.97
Model 2: Alternative Activation Functions
Changes: ELU and Tanh activation functions, lower dropout rate.
Evaluation:
MSE: 0.263
Accuracy: 0.80
ROC AUC: 0.98
Model 3: Expanded Architecture with Batch Normalization
Changes: Increased layer size, batch normalization.
Evaluation:
MSE: 1.143
Accuracy: 0.13
ROC AUC: 0.49
Optimizers and Early Stopping
Optimizers: Adam (Models 1 & 2) and SGD with momentum (Model 3).
Early Stopping: Monitors validation loss, halts training after 3 epochs without improvement.
Results
Summary of Model Performance
Model	MSE	Accuracy	Precision	Recall	F1 Score	ROC AUC
1	0.252	0.81	0.81	0.81	0.80	0.97
2	0.263	0.80	0.80	0.80	0.79	0.98
3	1.143	0.13	0.25	0.13	0.12	0.49
Conclusion: Model 1 exhibits the best balance between regression and classification performance.

Hyperparameter Tuning with Optuna
Parameters Tuned:

Learning Rate: Range from 1e-5 to 1e-1.
Dropout Rate: Range from 0.1 to 0.8.
Hidden Dimensions: Options of 64, 128, 256, 512 nodes.
Best Hyperparameters:

Learning Rate: 0.00047
Dropout Rate: 0.354
Hidden Layers: 512 nodes each
Final Model and Evaluation
The final model incorporates optimized hyperparameters identified through Optuna, achieving the following performance on the test dataset:

Metric	Value
MSE	0.249
Accuracy	0.83
Precision	0.82
Recall	0.83
F1 Score	0.82
ROC AUC	0.98
The model balances both regression and classification tasks effectively, yielding strong predictive accuracy with high generalization.

Note: The model parameters are saved in a .pth file for future inference or deployment.
