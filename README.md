# E-Commerce Customer Churn Prediction

This project aims to predict which customers are likely to unsubscribe (churn) from email campaigns using various machine learning models based on engagement metrics. The dataset contains customer information such as tenure, number of devices registered, preferred order category, satisfaction score, and other behavioral features. The goal is to predict churn and provide actionable insights to proactively target customers.

### Problem Statement

Retaining customers is more cost-effective than acquiring new ones. This project aimed to predict customer churn based on behavioral data to enable proactive retention strategies.

### Solution Approach

Data: Customer demographics, engagement metrics, and subscription history.

Methods:

1. Data Cleaning: Missing values were handled, outliers were removed from numerical columns, and categorical features were encoded using one-hot encoding.
2. Feature Engineering: Derived features like average cashback per interaction and complaints per tenure were added to improve predictive power.
3. Data Balancing: SMOTE and random undersampling were applied to balance the target variable classes.
4. Model Training: Various models were trained, including Logistic Regression, XGBoost, and Random Forest. Logistic Regression was adjusted for class imbalance, and the XGBoost model was trained with scale adjustments to handle imbalance.
4. Evaluation: Models were evaluated on accuracy, precision, recall, and F1-score. Additionally, the decision threshold for Logistic Regression was adjusted to optimize recall, providing a better balance between precision and recall.

## Dataset

The dataset is from an e-commerce company and contains customer behavior data. The target variable is `Churn`, a binary indicator of whether a customer has unsubscribed from email campaigns. The dataset contains information on customer demographics, engagement history, and transactional behavior. Some key features include:

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

## Class Distribution

After balancing the dataset to address class imbalance, the distribution was as follows:
- Non-Churn (0): 1934 samples
- Churn (1): 1354 samples

### Results

- Achieved an F1 score of 0.82 for churn prediction.
- Identified key churn drivers: low engagement frequency and long response times for support queries.
- Recommended targeted outreach campaigns, projected to reduce churn by 10%.

## Key Findings

- The XGBoost model demonstrated the highest accuracy in predicting customer churn, with an overall accuracy of 94%.
- The Logistic Regression model, while slightly less accurate, offered good interpretability and could be adjusted to prioritize recall.

### Future Work

1. Hyperparameter Tuning: Additional tuning of models like XGBoost and Random Forest to potentially improve performance.

2. Feature Engineering: Explore more derived features or interactions between features for improved predictions.

3. Explainability: Use SHAP values to explain model predictions in a more interpretable way, offering insights into key drivers of churn.

### Source: 

https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn


