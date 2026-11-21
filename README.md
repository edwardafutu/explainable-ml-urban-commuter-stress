# Explainable Machine Learning for Urban Commuter Stress

## Project Overview

This group project used machine learning to predict whether commuters experience high commuting stress. Using a simulated dataset of 500,000 respondents, we explored the factors associated with high commuter stress and compared different classification models.

The analysis used a stratified sample of 9,925 respondents for exploratory analysis and model development.

## Methods and Tools

- **Programming:** R
- **Modelling:** Elastic Net Logistic Regression, Support Vector Machine (SVM), XGBoost and Spark MLlib Logistic Regression.
- **Model evaluation:** Stratified cross-validation, ROC AUC and F1 score.
- **Explainability:** SHAP values, feature importance and Partial Dependence Plots (PDP).
- **Big data processing:** Apache Spark and sparklyr.

## Key Findings

- XGBoost achieved the highest test ROC AUC of **0.908**.
- Commute time, monthly income and number of transfers were among the most influential predictors.
- SHAP values and Partial Dependence Plots helped explain how individual predictors influenced model predictions.
- Spark logistic regression achieved an ROC AUC of **0.905** on the full simulated population of 500,000 respondents.

## My Contribution

I contributed to data exploration and led the XGBoost modelling and explainability component of the group project, including hyperparameter tuning, SHAP analysis and Partial Dependence Plots.

## Academic Context

- **Course:** BDAT 604 – Machine Learning Methods and Pattern Recognition
- **Institution:** University of Ghana
- **Academic Year:** 2025/2026
- **Project Type:** Group project
- **Group Members:** De-Graft Kofi Obeng, Edward Afutu Amoni and Samuel Opoku

*Note: The analysis used simulated data. The findings demonstrate machine learning methods and should not be interpreted as empirical evidence about actual commuter populations.*
