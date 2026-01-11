# Predicting Student Depression Using Regularized Logistic Regression

## Overview
This project develops and evaluates predictive models for student depression using
LASSO and Ridge logistic regression. The goal is to assess how academic, lifestyle,
and demographic factors contribute to depression risk while examining the impact
of dominant and sensitive predictors on model performance and interpretability.

The analysis compares models trained **with and without a highly influential predictor**
(suicidal thoughts) to evaluate robustness, ethical considerations, and coefficient
shrinkage behavior.

## Data
- Source: Publicly available Student Depression Dataset (Kaggle)
- Observations: 503 students
- Features include academic pressure, study satisfaction, sleep duration, dietary habits,
  financial stress, family history of mental illness, and demographic variables.
- Categorical variables were encoded numerically and all predictors were standardized.

## Methods
- Logistic regression with LASSO and Ridge regularization (`glmnet`)
- 10-fold cross-validation for optimal penalty (lambda) selection
- Performance evaluated using:
  - Accuracy (train/test)
  - ROC curves and AUC
  - McFadden’s pseudo R²
  - Confusion matrices
  - Coefficient paths and variable importance

## Key Findings
- Models including the dominant predictor achieved near-perfect classification performance.
- Removing the dominant predictor resulted in more realistic but still strong models,
  highlighting academic pressure and dietary habits as consistent predictors.
- LASSO demonstrated stronger variable selection and interpretability compared to Ridge,
  even when dominant predictors were excluded.

## Tools & Technologies
- R, RStudio
- glmnet, tidyverse, caret, pROC
- Statistical modeling, cross-validation, regularization

## Files
- `/code/`: Reproducible R scripts for data cleaning, modeling, and evaluation
- `/data/`: Cleaned dataset and data dictionary
- `/report/`: Full project report with methodology and results

## Author
Joshua Stewart-Roberts  
Rutgers University – New Brunswick  
Data Science
