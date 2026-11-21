# Explainable Machine Learning for Urban Commuter Stress

## Project Overview

This group project used machine learning to predict whether commuters experience high commuting stress. Using a simulated dataset of 500,000 respondents, we explored factors associated with high commuter stress and compared different classification models.

A stratified sample of 9,925 respondents was used for exploratory analysis and model development.

## Methods and Tools

- **Programming:** R
- **Modelling:** Elastic Net Logistic Regression, Support Vector Machine (SVM), XGBoost and Spark MLlib Logistic Regression.
- **Model evaluation:** Stratified cross-validation, ROC AUC and F1 score.
- **Explainability:** SHAP values, feature importance and Partial Dependence Plots (PDP).
- **Big data processing:** Apache Spark and sparklyr.

## Key Findings

- XGBoost achieved a test ROC AUC of **0.908**.
- Commute time, monthly income and number of transfers were among the influential predictors.
- SHAP values and Partial Dependence Plots helped explain how predictors influenced model predictions.
- Spark logistic regression achieved an ROC AUC of **0.905** using the full simulated population of 500,000 respondents.

## My Contribution

I contributed to data exploration and led the XGBoost modelling and explainability component of the group project, including hyperparameter tuning, SHAP analysis and Partial Dependence Plots.

## Repository Contents

| File | Description |
|---|---|
| `BDAT604_All_Tasks_1-6.R` | R script containing the data simulation, exploratory analysis, model development, explainability and model comparison tasks. |
| `BDAT604_Report_Tasks1-6_FFFs.pdf` | Project report documenting the methodology, analysis and findings. |

## Running the Analysis

The analysis script is designed to be run from top to bottom in a single R session because later tasks reuse objects created in earlier sections.

The script uses R packages including `tidyverse`, `tidymodels`, `glmnet`, `kernlab`, `xgboost`, `vip`, `DALEXtra`, `shapviz`, `sparklyr`, `lubridate` and `knitr`.

Apache Spark and a compatible Java environment are also required for the distributed modelling section.

**Note:** The script generates its simulated dataset internally. The project does not use an external commuter survey dataset.

## Academic Context

- **Course:** BDAT 604 – Machine Learning Methods and Pattern Recognition
- **Institution:** University of Ghana
- **Academic Year:** 2025/2026
- **Project Type:** Group project
- **Group Members:** De-Graft Kofi Obeng, Edward Afutu Amoni and Samuel Opoku

*The analysis uses simulated data. Its findings demonstrate machine learning methods and should not be interpreted as empirical evidence about actual commuter populations.*
