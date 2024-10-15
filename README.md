# E-Commerce Customer Churn Prediction

This project aims to predict which customers are likely to unsubscribe (churn) from email campaigns using various machine learning models based on engagement metrics. The dataset contains customer information such as tenure, number of devices registered, preferred order category, satisfaction score, and other behavioral features. The goal is to predict churn and provide actionable insights to proactively target customers.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Features](#features)
- [Project Workflow](#project-workflow)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [SHAP Explainability](#shap-explainability)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [License](#license)

## Project Overview

Churn prediction is critical for businesses to retain customers and offer targeted marketing efforts. In this project, we leverage customer engagement metrics to predict the likelihood of churn using machine learning algorithms such as Logistic Regression, Random Forest, and XGBoost.

## Dataset

The dataset is from an e-commerce company and contains customer behavior data. The target variable is `Churn`, a binary indicator of whether a customer has unsubscribed from email campaigns.

### Features:

- **Tenure**: How long the customer has been with the company (numeric)
- **WarehouseToHome**: Distance between the warehouse and the customer's home (numeric)
- **NumberOfDeviceRegistered**: Number of devices registered by the customer (numeric)
- **PreferedOrderCat**: Preferred order category (categorical)
- **SatisfactionScore**: Satisfaction score based on customer service (numeric)
- **MaritalStatus**: Marital status of the customer (categorical)
- **NumberOfAddress**: Number of addresses linked to the customer (numeric)
- **Complain**: Whether the customer has filed a complaint (binary)
- **DaySinceLastOrder**: Days since the customer last placed an order (numeric)
- **CashbackAmount**: Average cashback earned by the customer (numeric)
- **Churn**: Target variable (binary)

## Objective

- Predict customer churn based on engagement and demographic features.
- Identify the most important features influencing churn.
- Use SHAP (SHapley Additive exPlanations) to explain model decisions.
- Generate synthetic data to evaluate the model.

## Project Workflow

### Data Preprocessing

1. **Handling Missing Values**: Missing values in numeric columns such as `Tenure`, `WarehouseToHome`, and `DaySinceLastOrder` were filled with the median values.
2. **Outlier Removal**: Outliers were detected and removed using the Interquartile Range (IQR) method for key numerical columns.
3. **Encoding Categorical Variables**: Categorical variables such as `PreferedOrderCat` and `MaritalStatus` were one-hot encoded to convert them into numeric format.
4. **Feature Scaling**: StandardScaler was used to normalize features like `Tenure`, `WarehouseToHome`, and `SatisfactionScore` to ensure uniform scaling.

### Feature Engineering

- **AvgCashbackPerInteraction**: Derived as cashback amount divided by the number of devices registered.
- **ComplaintsPerTenure**: Number of complaints per tenure.
- **EngagementIntensity**: A custom feature calculated as a sum of `NumberOfDeviceRegistered`, `SatisfactionScore`, and `Complain` to measure the engagement level.

### Model Training

- **Logistic Regression**: The model was trained with class balancing using the `class_weight` parameter to account for the imbalance in churn vs non-churn customers.
- **Random Forest**: A Random Forest Classifier was trained to analyze feature importance.
- **XGBoost Classifier**: A powerful tree-based algorithm that was trained with `scale_pos_weight` to handle class imbalance.
- **SMOTE and Undersampling**: A hybrid approach was used to balance the dataset using **SMOTE** (Synthetic Minority Over-sampling Technique) and **Random Undersampling**.

### Model Evaluation

- **Confusion Matrix**: Evaluates the performance of the models by showing the True Positive, False Positive, True Negative, and False Negative predictions.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Threshold Adjustment**: The default decision threshold of 0.5 was adjusted to 0.4 for logistic regression to improve recall and precision.

### SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain the predictions made by the XGBoost model:
- **Global Feature Importance**: SHAP summary plots show the overall importance of features across all predictions.
- **SHAP Dependence Plot**: Visualizes how a single feature (e.g., `Tenure`) affects the prediction across different values.
- **Local Interpretation**: SHAP force plots explain individual predictions, providing insight into which features contributed most to the prediction for a single customer.

## Key Findings

1. **Feature Importance**: 
   - `Tenure`, `ComplaintsPerTenure`, and `CashbackAmount` are the most important features for predicting churn.
   - Derived features like `AvgCashbackPerInteraction` and `EngagementIntensity` also significantly contribute to the model’s decisions.
   
2. **XGBoost Performance**:
   - XGBoost provided the best results compared to Logistic Regression and Random Forest with a good balance between precision and recall.
   
3. **Synthetic Data Predictions**:
   - When generating synthetic data to simulate churn scenarios, the model accurately predicted high churn probabilities for certain customers, providing a useful tool for further testing.


---

### Part 2: Code Explanation and Insights

#### Code Overview:

The goal of the project is to predict customer churn, focusing on handling the class imbalance and ensuring feature importance is understood through SHAP. Here's a breakdown of the process:

1. **Data Loading and Cleaning**: The dataset is loaded, and missing values are filled with median values for numerical features. Outliers are removed using the IQR method to improve model performance.

2. **Feature Engineering**: 
   - Features like `AvgCashbackPerInteraction`, `ComplaintsPerTenure`, and `EngagementIntensity` are created to enhance predictive power.
   - Categorical variables like `PreferedOrderCat` and `MaritalStatus` are encoded using one-hot encoding.

3. **Balancing the Dataset**: 
   - The dataset is imbalanced (more non-churn customers than churn customers).
   - A combination of **SMOTE** and **Random Undersampling** is used to create a balanced dataset, improving model training.

4. **Model Training**: 
   - **Logistic Regression** is used first, with class weight balancing to handle the imbalanced dataset. 
   - **Random Forest** is used to identify feature importance.
   - **XGBoost Classifier** is the final model, fine-tuned using `scale_pos_weight` to adjust for class imbalance.
   
5. **Evaluation**:
   - Confusion matrix and classification reports provide key metrics such as precision, recall, and F1-score.
   - Adjusting the decision threshold (from 0.5 to 0.4) helps improve model performance.

6. **SHAP Explainability**:
   - SHAP values are computed to explain the XGBoost model's predictions.
   - Global SHAP summary plots show feature importance, with `Tenure`, `ComplaintsPerTenure`, and `CashbackAmount` being the most influential.
   - SHAP dependence plots and force plots provide both global and local model explainability.

#### Key Insights from the Output:

1. **Feature Importance**:
   - The most important features for predicting churn include:
     - **Tenure**: Longer-tenured customers are less likely to churn.
     - **ComplaintsPerTenure**: Customers with more complaints relative to their tenure are more likely to churn.
     - **CashbackAmount**: Higher cashback amounts have a positive impact on retention.
   - Engagement metrics, such as the number of devices registered and overall satisfaction score, are also important indicators.

2. **Model Performance**:
   - **XGBoost** outperformed the other models with strong recall and precision.
   - **Synthetic data** was used to test the model, showing that certain customer behaviors (e.g., short tenure, low cashback, and complaints) significantly increase churn likelihood.

3. **SHAP Interpretability**:
   - SHAP analysis revealed that **Tenure** and **ComplaintsPerTenure** play critical roles in predicting churn.
   - The SHAP force plot allows a deep dive into individual customer predictions, helping businesses understand why a specific customer is at risk of churning.


### Source:

https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn


