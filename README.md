# E-Commerce Customer Churn Prediction

This project aims to predict customer churn (unsubscribing from email campaigns) using machine learning models based on various engagement metrics. The goal is to identify customers at risk of churn and provide actionable insights to improve customer retention through targeted strategies.

### Problem Statement

Retaining existing customers is more cost-effective than acquiring new ones. By predicting customer churn based on engagement and behavioral data, businesses can take proactive measures to improve retention and customer loyalty.

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

### Solution Approach

Data: Customer demographics, engagement metrics, and subscription history.

### Methods

1. **Data Cleaning**: Missing values were handled, outliers were removed from numerical columns, and categorical features were encoded using one-hot encoding.
2. **Feature Engineering**: Derived features like average cashback per interaction and complaints per tenure were added to improve predictive power.
3. **Data Balancing**: SMOTE and random undersampling were applied to balance the target variable classes.
4. **Model Training**: Various models were trained, including Logistic Regression, XGBoost, and Random Forest. Logistic Regression was adjusted for class imbalance, and the XGBoost model was trained with scale adjustments to handle imbalance.
5. **Evaluation**: Models were evaluated on accuracy, precision, recall, and F1-score. Additionally, the decision threshold for Logistic Regression was adjusted to optimize recall, providing a better balance between precision and recall.

### Results

#### Model Performance
- The XGBoost model demonstrated the highest performance with an overall accuracy of 94%.
- The Logistic Regression model, though slightly less accurate, offered strong interpretability and high recall (0.95 for churn class), making it useful for identifying at-risk customers.

#### Key Drivers of Churn
- Customers with higher complaint frequency and shorter tenures are more likely to churn.
- Tenure was found to be the most influential factor, followed by complaints per tenure and cashback amount.

#### Churn Prediction Performance
- The model achieved an F1-score of 0.82 for churn prediction.
- Identified key churn drivers, such as low engagement frequency and long response times for support queries.

## Key Findings

- XGBoost: Best-performing model, with the highest accuracy of 94%.
- Logistic Regression: Offers a good balance of interpretability and performance, with the ability to adjust decision thresholds for better recall.
- Churn Drivers: Low engagement and poor support experiences are the most significant predictors of churn.

### Future Work

1. Hyperparameter Tuning: Additional tuning of models like XGBoost and Random Forest to potentially improve performance.

2. Feature Engineering: Explore more derived features or interactions between features for improved predictions.

3. Explainability: Use SHAP values to explain model predictions in a more interpretable way, offering insights into key drivers of churn.

### Source: 

Dataset: [E-commerce Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn)
