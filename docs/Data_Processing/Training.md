# Training Data Documentation

## Configuration: Splitting and Preparing the Training Data


## Overview:
- The training dataset consists of 5,634 rows and 20 columns. It is used to train the machine learning models in the Customer Churn Analysis project. The dataset contains various features that represent customer demographics, service usage patterns, and contract information. The goal is to predict whether a customer will churn based on these features.


## Dataset Structure:

- **SeniorCitizen:** 1 if the customer is a senior citizen, 0 otherwise

- **tenure:** time (in months) for which customer has stayed with the company

- **MonthlyCharges:** charges for each month that the customer has stayed

- **gender_Female, gender_Male:** binary indicator of gender of the customer

- **Dependents_No, Dependents_Yes:** binary indicator of whether customer has dependents

- **PhoneService_No, PhoneService_Yes:** Binary indicator of whether person has phone service or not

- **MultipleLines_No, MultipleLines_Yes:** whether the person has multiple phone lines

- **InternetService_DSL, InternetService_Fiber optic:** the kind of internet service

- **Contract_Month-to-month, Contract_One year, Contract_Two year:** the type of contract

- **Churn_No, Churn_Yes:** binary indicator if person churned

- **Charges_Per_Tenure:** Average charges per tenure

- **TotalCharges:** total of all charges.

## Summary:
- The training dataset is comprehensive, with a variety of features that capture customer behavior and characteristics. The preprocessing steps ensured that the data was clean and ready for modeling. The models trained on this dataset are expected to generalize well on unseen data, provided the feature distributions remain consistent.