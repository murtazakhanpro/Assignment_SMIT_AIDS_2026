# 🫀 Heart Failure Prediction – Classification Assignment

> **SMIT AIDS 2026 | Assignment 8b**  
> **Student:** Murtaza Ali  
> **Deadline:** 08-08-2026

---

## 📋 Project Overview

A complete machine-learning classification pipeline built on the **Heart Failure Clinical Records** dataset (UCI / Kaggle). The goal is to predict whether a patient will survive heart failure based on 12 clinical features.

---

## 📂 Repository Structure

```
Assignment_8b(Machine_learning_Classic_regression)/
├── Heart_Failure_Classification_Assignment.ipynb  ← Main Jupyter Notebook
├── heart_failure_clinical_records_dataset.csv      ← Dataset
├── Classification Assignment.md                    ← Original assignment brief
└── README.md                                       ← This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Heart Failure Clinical Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) |
| Rows | 299 |
| Features | 12 (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time) |
| Target | `DEATH_EVENT` (0 = Survived, 1 = Died) |

---

## 🔧 Pipeline Steps

| # | Step | Details |
|---|---|---|
| 1 | **Data Analysis** | Shape, dtypes, describe, info |
| 2 | **Missing Values** | None found — no imputation needed |
| 3 | **Encoding** | Not required — all features numeric |
| 4 | **Train-Test Split** | 80/20, stratified |
| 5 | **Feature Scaling** | `StandardScaler` (fit on train only) |
| 6 | **Class Imbalance** | SMOTE applied on training set |
| 7 | **Model Training** | 6 classifiers trained & compared |
| 8 | **Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report |
| 9 | **Hyperparameter Tuning** | `GridSearchCV` (5-fold CV) for RF & GB |
| 10 | **Feature Importance** | Top features from best Random Forest |

---

## 🤖 Models Used

- Logistic Regression
- Decision Tree
- Random Forest ⭐ (Best Base)
- Gradient Boosting ⭐ (Tuned)
- SVM
- KNN

---

## 🚀 How to Run

```bash
# Install dependencies
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn

# Launch notebook
jupyter notebook Heart_Failure_Classification_Assignment.ipynb
```

---

## 📈 Key Results

- **Best metric used:** F1-Score (most appropriate for imbalanced medical data)
- **Top features:** `time`, `serum_creatinine`, `ejection_fraction`
- SMOTE improved minority class (Died) Recall significantly
- Tuned ensemble models achieved the highest F1-Scores

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```
