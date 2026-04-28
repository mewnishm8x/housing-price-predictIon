# 🏠 AI Housing Price Prediction System

# Live Demo Link: 
# 📌 Project Overview

This project is a machine learning-based housing price prediction system that estimates real estate prices using structured property data. It applies multiple regression models and evaluates their performance to determine the most effective approach for prediction.

🎯 Objective

The goal of this project is to:

Predict housing prices based on key property features
Compare multiple regression models
Evaluate model performance using statistical metrics
Build a reliable and interpretable ML pipeline

📊 Dataset
Source: Kaggle USA Housing Dataset
Type: Structured tabular data
Size: ~5,000 samples
🔑 Features Used:
Average Area Income
House Age
Number of Rooms
Number of Bedrooms
Area Population

🧹 Data Preprocessing

The dataset was cleaned and prepared using the following steps:

Removed duplicate entries
Handled missing values using mean imputation
Dropped irrelevant or non-informative columns
Split data into training and testing sets (80/20 split)

🤖 Models Used

The following regression models were tested:

Linear Regression
Ridge Regression (L2 Regularization)
Lasso Regression (L1 Regularization)
Elastic Net
Random Forest Regressor (Final Model)

Final Model Choice
🌲 Random Forest Regressor

Chosen due to:

Strong predictive performance
Ability to capture non-linear relationships
Reduced risk of overfitting through ensemble learning

📈 Model Performance
Random Forest
Training R²: 0.9842
Testing R²: 0.8832
Training MAE: $35,293
Testing MAE: $94,021
Linear Models
Ridge R²: 0.9180
Lasso R²: 0.9180
Elastic Net R²: 0.9161

⚙️ Evaluation Metrics
R² Score: Measures how well the model explains variance in housing prices
MAE (Mean Absolute Error): Measures average prediction error in dollars

🔁 Machine Learning Pipeline
Data Collection (Kaggle dataset)
Data Cleaning & Preprocessing
Feature Selection
Train/Test Split (80/20)
Model Training
Prediction
Evaluation

🚀 Future Improvements
Hyperparameter tuning for better optimization
Cross-validation for more reliable evaluation
Feature engineering (ratios, interactions, etc.)
Deployment as a web application
Use of larger and region-specific datasets

🛠️ Tech Stack
Python
Pandas & NumPy
Scikit-learn
Matplotlib / Seaborn
Jupyter Notebook

👨‍💻 Author

Rashaad D Roberts
CAP4630-002 Intro to AI Intelligence
Spring 2026

📌 Notes:
Used Streamlit + Visual Studio Code for this project. 
