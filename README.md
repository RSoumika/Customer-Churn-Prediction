# Customer Churn Prediction System

## Overview

This project predicts whether a telecom customer is likely to churn using Machine Learning.

The objective is to identify customers at risk of leaving and help businesses improve customer retention strategies.

The solution includes:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning Model Development
* Model Evaluation
* Interactive Streamlit Web Application

---

## Dataset

The dataset contains:

* 7,043 customer records
* 21 customer attributes

Key features include:

* Contract Type
* Tenure
* Monthly Charges
* Internet Service
* Payment Method
* Customer Support Services

Target Variable:

* Churn (Yes / No)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib
* Matplotlib

---

## Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Random Forest Classifier

The final deployed model is:

**Logistic Regression with Class Balancing**

---

## Model Performance

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 73.35% |
| Recall   | 79.41% |
| F1 Score | 61.30% |

The optimized model successfully identifies a large percentage of customers likely to churn.

---

## Key Business Insights

* Month-to-month customers have the highest churn rate.
* Customers with low tenure are more likely to leave.
* Higher monthly charges are associated with increased churn.
* Fiber Optic customers show significantly higher churn rates.

---

## Project Structure

```text
Customer-Churn-Prediction
│
├── app
├── data
├── models
├── python
├── reports
├── screenshots
├── .streamlit
└── requirements.txt
```

---

## Author

Soumika R
