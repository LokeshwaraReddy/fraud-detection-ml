# ============================================
# Transaction Fraud Detection System
# Dataset: creditcard_2023.csv
# ============================================

# ======================
# 1. Import Libraries
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# ======================
# 2. Load Dataset
# ======================

df = pd.read_csv("creditcard_2023.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ======================
# 3. Basic Information
# ======================

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ======================
# 4. EDA - Fraud Analysis
# ======================

fraud_counts = df['Class'].value_counts()

print("\nClass Distribution")
print(fraud_counts)

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

fraud_percentage = (fraud_counts[1] / len(df)) * 100

print(f"\nFraud Transactions Percentage: {fraud_percentage:.4f}%")

# ======================
# 5. Feature Engineering
# ======================

# Transaction Amount Features
if 'Amount' in df.columns:
    df['Amount_Log'] = np.log1p(df['Amount'])

# Time-based Features
if 'Time' in df.columns:
    df['Hour'] = (df['Time'] // 3600) % 24

# Velocity Feature Example
# Number of transactions in a time window
if 'Time' in df.columns:
    df['Transaction_Velocity'] = df['Time'].diff().fillna(0)

# ======================
# 6. Feature Selection
# ======================

X = df.drop('Class', axis=1)
y = df['Class']

# ======================
# 7. Train-Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# 8. Scaling
# ======================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 9. Handle Imbalance using SMOTE
# ======================

print("\nBefore SMOTE")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE")
print(pd.Series(y_train_smote).value_counts())

# ======================
# 10. Model Training
# ======================

# -------- Random Forest --------

rf_model = RandomForestClassifier(
    n_estimators=10,
    max_depth=5,
    #class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# -------- XGBoost --------

xgb_model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# ======================
# 11. Predictions
# ======================

rf_preds = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:,1]

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:,1]

# ======================
# 12. Evaluation Function
# ======================

def evaluate_model(y_true, preds, probs, model_name):

    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    print(f"\n========== {model_name} ==========")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")

    print("\nClassification Report")
    print(classification_report(y_true, preds))

    print("\nConfusion Matrix")
    cm = confusion_matrix(y_true, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    RocCurveDisplay.from_predictions(y_true, probs)
    plt.title(f"{model_name} ROC Curve")
    plt.show()

# ======================
# 13. Model Evaluation
# ======================

evaluate_model(y_test, rf_preds, rf_probs, "Random Forest")

evaluate_model(y_test, xgb_preds, xgb_probs, "XGBoost")

# ======================
# 14. Feature Importance
# ======================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features")
print(feature_importance.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)
plt.title("Top 10 Important Features")
plt.show()

# ======================
# 15. Save Model
# ======================

import joblib

joblib.dump(rf_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")

# ============================================
import joblib

joblib.dump(rf_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")