# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:07:58 2025

@author: chinm
"""

# ------------------------------------------------------------
# TELCO CUSTOMER CHURN PREDICTION
# ------------------------------------------------------------

# ------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
# Make sure the CSV file is in the same directory
df = pd.read_csv(r"C:\Users\chinm\OneDrive\Pictures\Desktop\Python project\Telco Costumer churn prediction\Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
df.head()
df.info()
df.describe()

# ------------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------------

# Convert TotalCharges to numeric (it is stored as string)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing TotalCharges
df = df.dropna()

# Drop customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True, errors="ignore")

# ------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

# Churn distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Numerical features distribution
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.histplot(df["tenure"], kde=True)
plt.title("Tenure Distribution")

plt.subplot(1,3,2)
sns.histplot(df["MonthlyCharges"], kde=True)
plt.title("Monthly Charges Distribution")

plt.subplot(1,3,3)
sns.histplot(df["TotalCharges"], kde=True)
plt.title("Total Charges Distribution")

plt.tight_layout()
plt.show()

# Categorical vs Churn
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn vs Contract")

plt.subplot(1,3,2)
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn vs Internet Service")

plt.subplot(1,3,3)
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Churn vs Payment Method")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# DATA PREPROCESSING & FEATURE ENGINEERING
# ------------------------------------------------------------

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# ------------------------------------------------------------
#TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# LOGISTIC REGRESSION: FEATURE COEFFICIENTS
# ------------------------------------------------------------

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coef_df.head(10)
)
plt.title("Logistic Regression - Top Features Affecting Churn")
plt.show()


# ------------------------------------------------------------
# LOGISTIC REGRESSION SIGMOID CURVE (TENURE vs CHURN)
# ------------------------------------------------------------

# Use only one feature for visualization
X_single = df[["tenure"]]
y_single = df["Churn"]

# Scale the feature
scaler_single = StandardScaler()
X_single_scaled = scaler_single.fit_transform(X_single)

# Train logistic regression with single feature
lr_single = LogisticRegression()
lr_single.fit(X_single_scaled, y_single)

# Create smooth values for curve
tenure_range = np.linspace(
    X_single["tenure"].min(),
    X_single["tenure"].max(),
    300
).reshape(-1, 1)

tenure_range_scaled = scaler_single.transform(tenure_range)

# Predict probabilities
prob_curve = lr_single.predict_proba(tenure_range_scaled)[:, 1]

plt.figure(figsize=(8,6))

# Scatter actual data points
plt.scatter(
    X_single["tenure"],
    y_single,
    c=y_single,
    cmap="bwr",
    alpha=0.4,
    label="Actual Data"
)

# Plot sigmoid curve
plt.plot(
    tenure_range,
    prob_curve,
    color="black",
    linewidth=2,
    label="Logistic Regression Curve"
)

plt.xlabel("Tenure (months)")
plt.ylabel("Probability of Churn")
plt.title("Logistic Regression: Churn Predictive Curve (Tenure)")
plt.legend()
plt.show()


# ------------------------------------------------------------
# DECISION TREE VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(20,10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    max_depth=3
)
plt.title("Decision Tree Classifier (Top Levels)")
plt.show()


# ------------------------------------------------------------
# RANDOM FOREST: FEATURE IMPORTANCE
# ------------------------------------------------------------

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=rf_importance.head(10)
)
plt.title("Random Forest - Top Feature Importances")
plt.show()

# ------------------------------------------------------------
# XGBOOST: FEATURE IMPORTANCE
# ------------------------------------------------------------

xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=xgb_importance.head(10)
)
plt.title("XGBoost - Top Feature Importances")
plt.show()



# ------------------------------------------------------------
# MODEL EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob)
    }

# ------------------------------------------------------------
# PERFORMANCE COMPARISON TABLE
# ------------------------------------------------------------
results = pd.DataFrame({
    "Logistic Regression": evaluate_model(y_test, lr_pred, lr_prob),
    "Decision Tree": evaluate_model(y_test, dt_pred, dt_prob),
    "Random Forest": evaluate_model(y_test, rf_pred, rf_prob),
    "XGBoost": evaluate_model(y_test, xgb_pred, xgb_prob)
}).T

print("\nModel Performance Comparison:\n")
print(results)

# ------------------------------------------------------------
# MODEL PERFORMANCE COMPARISON - BAR GRAPH
# ------------------------------------------------------------

results.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------
models = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred
}

plt.figure(figsize=(14,4))

for i, (name, pred) in enumerate(models.items(), 1):
    plt.subplot(1,4,i)
    sns.heatmap(confusion_matrix(y_test, pred),
                annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FINAL SUMMARY (FOR REPORT / CODE COMMENT)
# ------------------------------------------------------------
"""
Final Conclusion:

Based on evaluation metrics such as Accuracy, F1-score, and ROC–AUC,
Logistic Regression performed best for this dataset. This indicates that
customer churn behavior in the given data follows largely linear patterns.
While ensemble models like Random Forest and XGBoost performed competitively,
they did not outperform the simpler baseline model.

This highlights that model performance depends on data characteristics,
and simpler, interpretable models can sometimes be more effective than
complex ensemble techniques.
"""
