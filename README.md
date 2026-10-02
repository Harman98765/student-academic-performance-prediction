# Student Academic Performance Prediction

## Overview

This project uses machine learning to predict a student's overall academic score based on study habits, attendance-related information, extracurricular activities, gender, part-time job status, and career aspiration.

The project includes exploratory data analysis, preprocessing, multiple regression models, hyperparameter tuning, error analysis, and a Streamlit web application.

## Features

- Exploratory Data Analysis
- Numerical and categorical feature preprocessing
- One-Hot Encoding
- Multiple regression models
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- Residual and error analysis
- Feature importance analysis
- Interactive Streamlit application

## Machine Learning Models

The following models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Gradient Boosting achieved the strongest test-set performance among the models evaluated.

## Final Model

The final model is a tuned Gradient Boosting Regressor.

Test-set performance:

- MAE: 3.66
- RMSE: 4.74
- R²: 0.477

## Input Features

The application uses:

- Gender
- Part-time job status
- Absence days
- Extracurricular activities
- Weekly self-study hours
- Career aspiration

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Project Structure

```text
student-performance-prediction/
│
├── Student_Academic_Performance_Prediction.ipynb
├── app.py
├── student_performance_model.pkl
├── requirements.txt
└── README.md 

## Dataset

The dataset used for this project was obtained from Kaggle.

The raw dataset is not included in this repository because it contains identifying fields that are not used by the machine learning model.

The target variable, `overall_score`, is calculated as the average of the seven subject scores.
