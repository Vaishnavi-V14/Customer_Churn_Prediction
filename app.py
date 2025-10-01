import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Customer Churn Dashboard",
                   layout="wide",
                   page_icon="📊")

st.title(" Customer Churn Dashboard")

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    customers = pd.read_csv("customer_data.csv")
    transactions = pd.read_csv("transaction_data.csv")
    return customers, transactions

customers, transactions = load_data()

# Encode categorical columns
le = LabelEncoder()
for col in customers.select_dtypes(include=["object"]).columns:
    customers[col] = le.fit_transform(customers[col].astype(str))

X = customers.drop("Churn", axis=1)
y = customers["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Sidebar Navigation
# ==============================
menu = ["📈 KPIs & Overview", "🔮 Churn Models", "🤖 ANN Training", "💰 Sales Trends", "👥 Segmentation", "📊 Insights"]
choice = st.sidebar.radio("Navigate Dashboard", menu)

# ==============================
# 1. KPIs & Overview
# ==============================
if choice == "📈 KPIs & Overview":
    st.header("📈 Key Business Metrics")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(customers)
    churn_rate = customers["Churn"].mean() * 100
    total_sales = transactions["amount"].sum()
    avg_sales = transactions["amount"].mean()

    col1.metric("👥 Total Customers", f"{total_customers}")
    col2.metric("⚠️ Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col4.metric("📊 Avg Sales", f"${avg_sales:,.0f}")

    # Churn distribution pie chart
    fig = px.pie(customers, names="Churn", title="Churn Distribution", hole=0.4,
                 color="Churn", color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔍 Feature Correlation Heatmap")
    corr = customers.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

# ==============================
# 2. Churn Models
# ==============================
elif choice == "🔮 Churn Models":
    st.header("🔮 Machine Learning Models for Churn Prediction")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        st.metric(label=f"{name} Accuracy", value=f"{acc:.2f}")

    # Comparison bar chart
    fig = px.bar(x=list(results.keys()), y=list(results.values()),
                 labels={"x": "Model", "y": "Accuracy"},
                 title="Model Comparison")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 3. ANN Training
# ==============================
elif choice == "🤖 ANN Training":
    st.header("🤖 Deep Learning Model (ANN)")

    ann = Sequential()
    ann.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
    ann.add(Dropout(0.3))
    ann.add(Dense(32, activation="relu"))
    ann.add(Dense(1, activation="sigmoid"))

    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = ann.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_split=0.2)

    loss, acc = ann.evaluate(X_test, y_test, verbose=0)
    st.metric("ANN Test Accuracy", f"{acc:.2f}")

    # Plot training history
    fig = px.line(y=[history.history["accuracy"], history.history["val_accuracy"]],
                  labels={"index": "Epoch", "value": "Accuracy"},
                  title="ANN Training Accuracy")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(legend=dict(title="Legend", itemsizing="constant"),
                      legend_traceorder="normal")
    fig.data[0].name = "Train Accuracy"
    fig.data[1].name = "Validation Accuracy"
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 4. Sales Trends
# ==============================
elif choice == "💰 Sales Trends":
    st.header("💰 Sales Trend Analysis")

    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["month"] = transactions["date"].dt.to_period("M")
    monthly_sales = transactions.groupby("month")["amount"].sum()

    # Interactive Plotly line chart
    fig = px.line(monthly_sales, x=monthly_sales.index.astype(str), y=monthly_sales.values,
                  labels={"x": "Month", "y": "Sales Amount"},
                  title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Top products (if product column exists)
    if "product" in transactions.columns:
        fig = px.bar(transactions.groupby("product")["amount"].sum().reset_index(),
                     x="product", y="amount", title="Top Selling Products")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# 5. Segmentation
# ==============================
elif choice == "👥 Segmentation":
    st.header("👥 Customer Segmentation (K-Means)")

    seg_features = customers.drop("Churn", axis=1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customers["Segment"] = kmeans.fit_predict(seg_features)

    fig = px.scatter(customers, x="Age", y="Balance", color="Segment",
                     title="Customer Segmentation by Age vs Balance",
                     color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Segmented Data",
                       customers.to_csv(index=False).encode("utf-8"),
                       file_name="segmented_customers.csv",
                       mime="text/csv")

# ==============================
# 6. Insights
# ==============================
elif choice == "📊 Insights":
    st.header("📊 Business Insights")
    st.write("""
    ✅ ANN detects **complex churn patterns** better than classical ML.  
    ✅ Random Forest & XGBoost are stable, interpretable models.  
    ✅ Sales analysis shows **seasonal revenue fluctuations**.  
    ✅ Segmentation enables **targeted retention strategies**.  
    ✅ Combining ML + ANN + Sales = **Full 360° Customer Intelligence**.  
    """)
