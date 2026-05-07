# Credit Card Fraud Detection

## Problem Statement
Credit card fraud causes billions in losses every year. This project builds a machine learning system to automatically detect fraudulent credit card transactions from a dataset with extreme class imbalance (only 0.17% fraud cases).

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | Only 492 fraud (0.17%)
- Features V1-V28 are PCA transformed for confidentiality
- Features: Amount, Time + 28 PCA components

## Approach
1. Exploratory Data Analysis (EDA) - analyzed amount and time distributions across fraud vs normal
2. Handled extreme class imbalance using SMOTE (227k → balanced dataset)
3. Built and compared 3 models
4. Evaluated using ROC-AUC, Precision, Recall, F1

## Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.97 |
| Random Forest | 0.98 |
| XGBoost | 0.99 |

**Best Model: XGBoost with ROC-AUC of 0.99**

### Single Transaction Prediction (XGBoost):
| Transaction Type | Prediction | Fraud Probability |
|---|---|---|
| Normal | ✅ Normal | 0.00% |
| Fraud | 🚨 Fraud | 100.00% |

## Key Findings
- Accuracy is completely misleading for imbalanced data — a model predicting all Normal gets 99.83% accuracy but catches zero fraud
- Features V17, V14, V12, V10 showed highest correlation with fraud in EDA
- SMOTE increased fraud training samples from 394 to 227,451 (balanced with normal)
- XGBoost outperformed all models with perfect fraud detection on test cases
- Recall is prioritized over precision — missing a fraud is far worse than a false alarm

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn

## Project Structure