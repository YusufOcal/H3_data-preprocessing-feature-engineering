# H3_data-preprocessing-feature-engineering

This project focuses on comprehensive data preprocessing and feature engineering techniques using Python. The primary emphasis is on handling missing values, encoding categorical variables, and standardizing/normalizing numerical features.

## Project Overview

The Heart Disease UCI dataset is used to demonstrate a complete data preprocessing pipeline, from initial data exploration to preparing machine learning-ready features. This project showcases various transformation techniques that form the foundation of effective machine learning models.

## Dataset Description

The Heart Disease UCI dataset includes several clinical features:

- **id**: Patient identifier
- **age**: Age of the patient in years
- **sex**: Gender of the patient (Male/Female)
- **dataset**: Source of the data (Cleveland, Hungary, Switzerland, VA Long Beach)
- **cp**: Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
- **trestbps**: Resting blood pressure in mm Hg
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (True/False)
- **restecg**: Resting electrocardiographic results
- **thalch**: Maximum heart rate achieved
- **exang**: Exercise induced angina (True/False)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (normal, fixed defect, reversable defect)
- **num**: Diagnosis of heart disease (0-4, where 0 = no presence, 1-4 = presence with increasing severity)

## Core Files

- `Hafta_3_Odev.ipynb`: **Main assignment notebook** containing:
  - Complete data exploration and cleaning workflow
  - Missing value analysis and handling strategies
  - Comprehensive implementation of various encoding techniques
  - Numerical feature standardization and normalization
  - Complete data preprocessing pipeline

- Supporting files:
  - `heart_disease_uci_case.csv`: Heart disease dataset with 920 patient records
  - `Odev.txt`: Detailed assignment requirements and steps

## Key Techniques Demonstrated

The assignment notebook showcases several essential data preprocessing skills:

### 1. Data Exploration and Cleaning
```python
import pandas as pd
df = pd.read_csv('heart_disease_uci.csv')
df.info()
df.describe()
```

### 2. Identifying Feature Types
```python
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
```

### 3. Missing Value Treatment
- Numerical features: Filled with appropriate measures of central tendency
```python
def fill_numerical_columns(df, numerical_columns):
    for column in numerical_columns:
        df[column].fillna(df[column].mean(), inplace=True)
    return df
```
- Categorical features: Filled with mode (most frequent values)
```python
def fill_categorical_columns(df, categorical_columns):
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df
```

### 4. Feature Scaling and Normalization
- **Min-Max Scaling**: Transforms features to [0,1] range
- **Z-Score Standardization**: Transforms features to mean=0, std=1
- **Robust Scaling**: Scaling resistant to outliers
- **Logarithmic Transformation**: For handling skewed distributions

### 5. Categorical Encoding Methods
- **Label Encoding**: Converting categories to sequential numbers
- **One-Hot Encoding**: Creating binary columns for each category
- **Binary Encoding**: Efficient encoding for high-cardinality features
- **Ordinal Encoding**: For variables with inherent ordering
- **Frequency Encoding**: Based on category frequency
- **Target Encoding**: Using relationship with target variable

### 6. Dummy Variable Trap Avoidance
Handling multicollinearity issues in one-hot encoded features by dropping the first category:
```python
pd.get_dummies(df, columns=['categorical_feature'], drop_first=True)
```

## Key Insights

The analysis reveals several important preprocessing considerations:
- The Heart Disease UCI dataset contains significant missing values in features like 'ca' and 'thal'
- Different encoding strategies lead to different feature representations
- Feature scaling impacts the range and distribution of numerical features
- Appropriate preprocessing techniques depend on the underlying data distribution and model requirements

## Technical Challenges

- **Handling Missing Data**: Implementing appropriate imputation strategies based on feature characteristics
- **Feature Type Selection**: Determining which encoding method works best for different categorical variables
- **Dummy Variable Trap**: Understanding and avoiding multicollinearity in encoded features
- **Data Transformation**: Selecting appropriate scaling methods based on data distributions

## Requirements

- Python 3.x
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook/Google Colab

## When to Use Different Techniques

- **Min-Max Scaling**: When distribution doesn't matter and you need values between 0-1
- **Z-Score Standardization**: When normal distribution is preferred with mean=0, std=1
- **Robust Standardization**: When dealing with datasets containing outliers
- **Logarithmic Transformation**: When dealing with skewed distributions with large value differences

- **Label Encoding**: For ordinal categorical data with inherent order
- **One-Hot Encoding**: For nominal categorical data without order relationship
- **Binary Encoding**: For high-cardinality nominal features to reduce dimensionality
- **Frequency Encoding**: When category frequency provides meaningful information

## Learning Outcomes

By working through this project, you'll gain practical experience in:
- Building complete data preprocessing pipelines
- Implementing different encoding techniques for categorical features
- Applying appropriate scaling methods for numerical features
- Handling missing values with appropriate strategies
- Understanding the impact of preprocessing on downstream machine learning tasks 