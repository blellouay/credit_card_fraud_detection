# Credit Card Fraud Detection using Machine Learning

## Overview

This project aims to develop a machine learning system capable of detecting fraudulent credit card transactions. Due to the highly imbalanced nature of fraud detection datasets, special attention is given to data preprocessing, feature engineering, class imbalance handling, model selection, hyperparameter tuning, and threshold optimization.

The project follows a complete end-to-end machine learning workflow, from exploratory data analysis to model evaluation and interpretation.

---

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset available on Kaggle:

* Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Transactions: 284,807
* Fraudulent transactions: 492
* Fraud ratio: approximately 0.17%

### Features

* `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
* `Amount`: Transaction amount.
* `V1` – `V28`: PCA-transformed features generated to preserve confidentiality.
* `Class`:

  * `0`: Legitimate transaction
  * `1`: Fraudulent transaction

---

## Project Structure

```text
credit_card_fraud_detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── best_model.pkl
│   ├── best_tuned_model.pkl
│   └── final_model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── Training.ipynb
│   ├── Hyperparameter_Tuning.ipynb
│   ├── Threshold_Optimization.ipynb
│   └── Final_Evaluation.ipynb
│
├── results/
│   ├── model_cv_results.csv
│   ├── feature_importance.csv
│   └── final_model_metrics.csv
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

The exploratory analysis focused on:

* Class imbalance analysis
* Duplicate detection and removal
* Missing value verification
* Transaction amount distribution
* Log transformation of transaction amounts
* Time-based fraud analysis
* Correlation analysis
* Feature distribution comparison by class

### Key Findings

* The dataset is extremely imbalanced.
* Fraudulent transactions represent less than 0.2% of all observations.
* Features such as V14, V10, V12, V11, and V17 show strong separation between fraudulent and legitimate transactions.
* Time-related features exhibit some variation but are not dominant predictors.

---

## Feature Engineering

Several preprocessing pipelines were evaluated:

### Raw Pipeline

* Original features only.

### Log Amount Pipeline

* Applied logarithmic transformation to transaction amounts:

```math
\log(Amount + 1)
```

### Cyclical Time Pipeline

Time was transformed into cyclical features:

```math
Hour_{sin} = \sin\left(\frac{2\pi Hour}{24}\right)
```

```math
Hour_{cos} = \cos\left(\frac{2\pi Hour}{24}\right)
```

### Scaled Pipeline

* Standardization applied for models sensitive to feature scales.

---

## Class Imbalance Handling

Fraud detection suffers from severe class imbalance.

To address this issue:

* SMOTE (Synthetic Minority Oversampling Technique) was applied.
* SMOTE was integrated inside an `imblearn.pipeline.Pipeline`.
* Oversampling was performed only during training folds to avoid data leakage.

---

## Models Evaluated

### Logistic Regression

Used as a baseline model.

### Random Forest

Tree-based ensemble model capable of capturing nonlinear relationships.

### XGBoost

Gradient boosting model optimized for tabular datasets.

---

## Cross-Validation Results

| Model               | AUPRC | ROC-AUC | Precision | Recall | F1    |
| ------------------- | ----- | ------- | --------- | ------ | ----- |
| Random Forest       | 0.846 | 0.980   | 0.910     | 0.812  | 0.857 |
| XGBoost             | 0.821 | 0.980   | 0.577     | 0.839  | 0.683 |
| Logistic Regression | 0.755 | 0.979   | 0.054     | 0.907  | 0.102 |

### Best Model

Random Forest achieved the highest AUPRC and was selected for further optimization.

---

## Hyperparameter Tuning

Hyperparameter optimization was performed using:

* RandomizedSearchCV
* 5-Fold Stratified Cross Validation
* AUPRC as the optimization metric

### Tuned Parameters

```python
{
    'model__n_estimators': 500,
    'model__min_samples_split': 10,
    'model__min_samples_leaf': 1,
    'model__max_features': 'sqrt',
    'model__max_depth': 20,
    'model__bootstrap': False
}
```

---

## Threshold Optimization

Instead of using the default threshold:

```python
probability >= 0.50
```

multiple thresholds were evaluated.

### Selected Threshold

```python
0.65
```

This threshold provided the best balance between:

* Precision
* Recall
* F1-score

---

## Final Results

### Test Set Performance

| Metric    | Score |
| --------- | ----- |
| AUPRC     | 0.802 |
| ROC-AUC   | 0.977 |
| Precision | 0.933 |
| Recall    | 0.737 |
| F1-score  | 0.824 |

### Interpretation

The final model successfully detects approximately 74% of fraudulent transactions while maintaining a precision above 93%, meaning that most transactions flagged as fraud are genuinely fraudulent.

This balance significantly reduces false alarms while still capturing the majority of fraudulent activity.

---

## Feature Importance

The most influential variables include:

1. V14
2. V10
3. V12
4. V11
5. V4
6. V17
7. V3
8. V7
9. V16
10. V9

These PCA-derived features contain the strongest fraud-related information within the dataset.

---

## Technologies Used

### Programming Language

* Python 3.14

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Imbalanced-learn
* XGBoost

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* VS Code

---

## Installation

Clone the repository:

```bash
git clone https://github.com/blellouay/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

Create a virtual environment:

```bash
python -m venv credit
```

Activate it:

```bash
credit\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

* SHAP explainability analysis
* MLflow experiment tracking
* Cost-sensitive learning
* Advanced ensemble methods
* Real-time fraud detection pipeline
* Model deployment using FastAPI and Docker

---

## Author

**Louay Blel**

Data Science Engineering Student

ESSAI – Higher School of Statistics and Information Analysis

Interested in Machine Learning, Artificial Intelligence, and Applied Data Science.
