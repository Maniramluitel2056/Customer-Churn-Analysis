# Testing Data Documentation

## Configuration: Data splitting & preparation Testing

## Overview
The testing dataset consists of 1,409 rows and 20 columns. It is used to evaluate the performance of the machine learning models trained on the training dataset. The testing dataset has the same structure as the training dataset and is used to simulate the model's performance on unseen data.


## Dataset Structure:
The testing dataset has the same columns and structure as the training dataset, ensuring consistency between training and evaluation. The features included are:

- **SeniorCitizen**
- **tenure**
- **MonthlyCharges**
- **gender_Female and gender_Male**
- **Dependents_No and Dependents_Yes**
- **PhoneService_No and PhoneService_Yes**
- **MultipleLines_No and MultipleLines_Yes**
- **InternetService_DSL and InternetService_Fiber optic**
- **Contract_Month-to-month, Contract_One year, and Contract_Two year**
- **Churn_No and Churn_Yes**
- **Charges_Per_Tenure**
- **TotalCharges**


## Summary:
The testing dataset is very important for how we evaluate a model. It offers an unbiased evaluation of the performance capabilities for such models under real-world scenarios. And to be able measure the model’s real performance for this new data, we kept a subset of our training set as testing part.

