# Credit Card Fraud Detection
## Problem Statement
Credit card fraud is a major problem for banks and customers. This project builds a machine learning model to detect fraudulent transactions from a highly imbalanced dataset.

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | Only 492 fraud (0.17%)
- Features V1-V28 are PCA transformed for confidentiality

## Approach
1. Exploratory Data Analysis (EDA)
2. Handling class imbalance using SMOTE
3. Built and compared 3 models:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Evaluated using ROC-AUC, Precision, Recall

## Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.97 |
| Random Forest | 0.98 |
| XGBoost | 0.99 |

> XGBoost performed best with ROC-AUC of 0.99

## Key Learnings
- Accuracy is misleading for imbalanced datasets
- SMOTE helps balance classes without losing data
- Recall is more important than precision for fraud detection

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```