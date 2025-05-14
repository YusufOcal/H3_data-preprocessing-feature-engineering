# Data Preprocessing and Feature Engineering Mastery

This project explores essential data preprocessing and feature engineering techniques with comprehensive Python implementations, focusing on practical applications for machine learning pipelines.

## Project Overview

The notebooks provide hands-on demonstrations of critical data preparation concepts that form the foundation of successful machine learning models:
- Advanced data cleaning and transformation workflows
- Statistical techniques for handling missing values
- Comprehensive feature scaling and normalization
- Industry-standard categorical encoding approaches

## Core Jupyter Notebooks

### `Hafta_3_Odev.ipynb` (Main Assignment)
The primary notebook implements a complete preprocessing pipeline with detailed explanations of:

- **Interpolation Techniques**: Thorough explanation of when interpolation is needed versus when it's not needed for quartile positions, with practical examples
- **Data Standardization Methods**:
  - Min-Max Scaling (Normalization): Implementation with `MinMaxScaler`
  - Z-Score Standardization: Implementation with `StandardScaler`
  - Robust Scaling: Implementation with `RobustScaler` for outlier resistance
  - Logarithmic Transformation: For handling skewed distributions

- **Categorical Encoding Methods**:
  - Label Encoding: For ordinal data
  - One-Hot Encoding: Using both scikit-learn and pandas approaches
  - Binary Encoding: Efficient alternative to one-hot for high-cardinality features
  - Ordinal Encoding: For variables with inherent ordering
  - Frequency Encoding: Based on category frequency
  - Target Encoding: Using relationship with target variable

### `3_hafta.ipynb` (Supporting Material)
Supplements the main assignment with additional explanations on:
- Fundamental pandas operations and methods for data manipulation
- In-depth theoretical explanation of normal vs. skewed distributions
- Decision guidance for selecting appropriate standardization techniques
- Visual comparisons of different transformation results

## Implementation Highlights

The notebooks demonstrate practical code implementations including:

```python
# Example: Comprehensive data exploration function
def examine_dataset(df):
  print('Dataset First 5 Rows')
  display(df.head())
  print('Dataset Information')
  display(df.info())
  print('Descriptive Statistics')
  display(df.describe().T)
```

```python
# Example: Data standardization with scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Z-Score standardization
scaler = StandardScaler()
z_scaled = scaler.fit_transform(df[numerical_cols])

# Min-Max scaling
minmax_scaler = MinMaxScaler()
normalized = minmax_scaler.fit_transform(df[numerical_cols])
```

```python
# Example: Categorical encoding with multiple techniques
# One-Hot Encoding
one_hot_df = pd.get_dummies(data, columns=['Category'], drop_first=True)

# Target Encoding
target_encoder = ce.TargetEncoder(cols=['Category'])
target_encoded = target_encoder.fit_transform(X['Category'], y)
```

## When to Use Different Techniques

The notebooks provide clear guidance on selecting techniques based on:
- Data distribution (normal vs. skewed)
- Presence of outliers
- Feature relationships
- Model requirements (distance-based vs. tree-based)

## Key Learning Outcomes

By working through these notebooks, you'll gain practical experience in:
- Selecting appropriate preprocessing techniques based on data characteristics
- Implementing complete preprocessing pipelines using industry-standard libraries
- Understanding the impact of different transformations on model performance
- Following best practices for production-ready feature engineering

This project serves as both a practical implementation guide and a reference for essential preprocessing techniques that form the foundation of effective machine learning models. 