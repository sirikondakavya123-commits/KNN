import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------- UI --------------------
st.title("🔍 KNN Classification App")
st.write("Interactive K-Nearest Neighbors classifier using synthetic data")

# Sidebar controls
st.sidebar.header("Model Parameters")

n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of Features", 2, 5, 3)
k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)

# -------------------- Data Generation --------------------
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------- Model --------------------
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

# -------------------- Results --------------------
st.subheader("📊 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# -------------------- Visualization --------------------
if n_features >= 2:
    st.subheader("📈 Data Visualization (First 2 Features)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)
