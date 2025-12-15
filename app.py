import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("📊 Telco Customer Churn Analysis Dashboard")
st.write("Simple dashboard to understand customer churn patterns and model performance.")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔍 Filters")

contract_type = st.sidebar.multiselect(
    "Select Contract Type",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_service = st.sidebar.multiselect(
    "Select Internet Service",
    df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

filtered_df = df[
    (df["Contract"].isin(contract_type)) &
    (df["InternetService"].isin(internet_service))
]

# -----------------------------
# KPIs
# -----------------------------
total_customers = filtered_df.shape[0]
churned = filtered_df[filtered_df["Churn"] == "Yes"].shape[0]
churn_rate = (churned / total_customers) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

# -----------------------------
# Churn distribution
# -----------------------------
st.subheader("📌 Churn Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Churn", data=filtered_df, ax=ax)
st.pyplot(fig)

# -----------------------------
# Churn vs Contract
# -----------------------------
st.subheader("📌 Churn vs Contract Type")

fig, ax = plt.subplots()
sns.countplot(x="Contract", hue="Churn", data=filtered_df, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# -----------------------------
# Monthly charges vs churn
# -----------------------------
st.subheader("📌 Monthly Charges vs Churn")

fig, ax = plt.subplots()
sns.boxplot(x="Churn", y="MonthlyCharges", data=filtered_df, ax=ax)
st.pyplot(fig)

# -----------------------------
# Model performance section
# -----------------------------
st.subheader("🤖 Model Performance Summary")

st.write("""
- Logistic Regression: Baseline and interpretable  
- Decision Tree: Easy to understand but prone to overfitting  
- Random Forest: Strong and stable performance  
- **XGBoost: Best performing model with highest ROC-AUC and F1-score**
""")

st.success("✅ XGBoost is the recommended model for churn prediction")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Created for Telco Customer Churn Prediction Project")
