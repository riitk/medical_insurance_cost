# medical_insurance_cost
Medical Insurance Cost Predicton Using ML

# Medical Insurance Cost Prediction

## Overview
This project aims to predict medical insurance costs based on various attributes such as age, sex, BMI, number of children, smoking status, and region. The dataset is analyzed and processed, and multiple machine learning models are trained and evaluated to determine the most accurate prediction model.

## Dataset
The dataset contains 1,338 rows with the following columns:
- `age`: Age of the individual
- `sex`: Gender of the individual (male, female)
- `bmi`: Body Mass Index
- `children`: Number of children/dependents covered by insurance
- `smoker`: Smoking status of the individual (yes, no)
- `region`: Residential area of the individual (northeast, northwest, southeast, southwest)
- `charges`: Medical insurance cost

## Data Preprocessing
1. **Basic Analysis**:
   - Checked for null values, dataset shape, and descriptive statistics using `df.info()` and `df.describe()`.
2. **Data Visualization**:
   - Used count plots, histograms, and scatter plots to study relationships between columns.
3. **Encoding Categorical Variables**:
   - Mapped `sex` to binary values: 
     ```python
     df["sex"] = df['sex'].map({"male": 0, "female": 1})
     ```
   - Mapped `smoker` to binary values: 
     ```python
     df["smoker"] = df['smoker'].map({"no": 0, "yes": 1})
     ```
   - One-hot encoded `region` and dropped the original column:
     ```python
     df = df.join(pd.get_dummies(df["region"], drop_first=True, dtype=int))
     df.drop("region", axis=1, inplace=True)
     ```
4. **Correlation Matrix**:
   - Plotted a heatmap to check correlations between features:
     ```python
     sns.heatmap(df.corr(), annot= True, fmt=".2f")
     ```

## Model Training
Performed train-test split and trained three regression models:
1. **Linear Regression**
2. **Lasso Regression**
3. **Support Vector Regressor (SVR)**

## Model Evaluation
Evaluated models using Mean Squared Error (MSE) and R-squared (R²) Score:
- **Linear Regression**: R² Score = 0.77316
- **Lasso Regression**: R² Score = 0.77318
- **SVR**: R² Score = -0.092 (poor performance)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/riitk/medical_insurance_cost.git
   cd medical_insurance_cost
