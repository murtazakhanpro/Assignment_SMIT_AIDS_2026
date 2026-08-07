
# ============================================================
# Heart Failure Prediction – Classification Assignment
# SMIT AIDS 2026 | Assignment 8b
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

print("All libraries imported successfully!")

# ── 2. LOAD DATASET ──────────────────────────────────────────
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(f"Dataset shape: {df.shape}")
df.head()

# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────
print("=== Dataset Info ===")
df.info()
print("\n=== Statistical Summary ===")
df.describe()

# Missing values check
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Target distribution
print("\n=== Target Distribution ===")
print(df['DEATH_EVENT'].value_counts())
print(f"Class Imbalance Ratio: {df['DEATH_EVENT'].value_counts(normalize=True).round(3).to_dict()}")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['DEATH_EVENT'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'], edgecolor='black')
axes[0].set_title('Class Distribution (Before SMOTE)')
axes[0].set_xticklabels(['Survived (0)', 'Died (1)'], rotation=0)
axes[0].set_ylabel('Count')
df['DEATH_EVENT'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                       colors=['steelblue', 'tomato'], startangle=90)
axes[1].set_title('Class Distribution Pie Chart')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# ── 4. PREPROCESSING ─────────────────────────────────────────
# No missing values → skip imputation
# All features are already numeric → no encoding needed

# Separate features and target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Train-test split (stratified to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Feature scaling applied using StandardScaler.")

# ── 5. HANDLE CLASS IMBALANCE WITH SMOTE ─────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE – Class counts: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ── 6. TRAIN MULTIPLE CLASSIFIERS ────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
    'SVM':                 SVC(random_state=42),
    'KNN':                 KNeighborsClassifier(),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='accuracy')
    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred),
        'CV Mean':   cv_scores.mean(),
        'CV Std':    cv_scores.std(),
    }
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

# ── 7. PERFORMANCE COMPARISON TABLE ──────────────────────────
results_df = pd.DataFrame(results).T.round(4)
print("\n=== Model Performance Comparison ===")
print(results_df.to_string())

# Bar chart comparison
results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', figsize=(14, 6), colormap='Set2', edgecolor='black')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=25, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ── 8. SELECT BEST MODEL ─────────────────────────────────────
best_model_name = results_df['F1-Score'].idxmax()
best_model      = models[best_model_name]
print(f"\nBest Model (by F1-Score): {best_model_name}")

# Confusion Matrix for best model
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Survived', 'Died'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix – {best_model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=150)
plt.show()

# ── 9. HYPERPARAMETER TUNING ─────────────────────────────────
# Tune Random Forest (usually the top performer)
print("\n=== Hyperparameter Tuning: Random Forest ===")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf,
                       cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_rf.fit(X_train_res, y_train_res)
print(f"Best RF Params: {grid_rf.best_params_}")
print(f"Best RF CV F1:  {grid_rf.best_score_:.4f}")

# Tune Gradient Boosting
print("\n=== Hyperparameter Tuning: Gradient Boosting ===")
param_grid_gb = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth':     [3, 5],
}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb,
                       cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_gb.fit(X_train_res, y_train_res)
print(f"Best GB Params: {grid_gb.best_params_}")
print(f"Best GB CV F1:  {grid_gb.best_score_:.4f}")

# ── 10. FINAL MODEL EVALUATION ───────────────────────────────
print("\n=== Final Tuned Models – Test Set Performance ===")
final_results = {}
for label, gs in [('RF (Tuned)', grid_rf), ('GB (Tuned)', grid_gb)]:
    y_pred_f = gs.best_estimator_.predict(X_test_scaled)
    final_results[label] = {
        'Accuracy':  accuracy_score(y_test, y_pred_f),
        'Precision': precision_score(y_test, y_pred_f),
        'Recall':    recall_score(y_test, y_pred_f),
        'F1-Score':  f1_score(y_test, y_pred_f),
    }
    print(f"\n── {label} ──")
    print(classification_report(y_test, y_pred_f, target_names=['Survived', 'Died']))

final_df = pd.DataFrame(final_results).T.round(4)
print("\n=== Final Comparison: Tuned Models ===")
print(final_df.to_string())

# ── 11. FEATURE IMPORTANCE ───────────────────────────────────
best_rf = grid_rf.best_estimator_
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importances – Tuned Random Forest')
plt.ylabel('Importance Score')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\n✅ Assignment Complete! All steps executed successfully.")
