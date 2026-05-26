# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import joblib

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Page configuration
st.set_page_config(
    page_title="KNN Classification",
    layout="wide"
)

# Custom styling
st.markdown(
    """
    <style>
    h1{
        text-align:center;
    }

    .stButton>button{
        width:100%;
        height:50px;
        background-color:#4CAF50;
        color:white;
        border:none;
        border-radius:10px;
        font-size:18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("House Price Category Prediction Using KNN")

# Download dataset directly from Kaggle
path = kagglehub.dataset_download(
    "shree1992/housedata"
)

# Read dataset
df = pd.read_csv(f"{path}/data.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset shape
st.write("Dataset Shape :", df.shape)

# Remove duplicates
df = df.drop_duplicates()

# Drop unwanted columns
drop_cols = ["date", "street", "country"]

# Drop columns
for col in drop_cols:

    # Check column exists
    if col in df.columns:

        # Drop column
        df = df.drop(col, axis=1)

# Create classification target
median_price = df["price"].median()

# Create price category
df["price_category"] = np.where(
    df["price"] >= median_price,
    1,
    0
)

# Drop original price column
df = df.drop("price", axis=1)

# Store encoders
encoders = {}

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill numerical missing values
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill categorical missing values
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical columns
for col in cat_cols:

    # Create encoder
    le = LabelEncoder()

    # Encode values
    df[col] = le.fit_transform(df[col])

    # Store encoder
    encoders[col] = le

# Correlation matrix
st.subheader("Correlation Matrix")

# Select only numerical columns
numeric_df = df.select_dtypes(include=np.number)

# Create correlation matrix
corr = numeric_df.corr()

# Create figure
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plot heatmap
heatmap = ax1.imshow(corr)

# Add labels
ax1.set_xticks(range(len(corr.columns)))
ax1.set_yticks(range(len(corr.columns)))

# Column names
ax1.set_xticklabels(corr.columns, rotation=90)
ax1.set_yticklabels(corr.columns)

# Add colorbar
plt.colorbar(heatmap)

# Show graph
st.pyplot(fig1)

# Features
X = df.drop("price_category", axis=1)

# Target
y = df["price_category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale dataset
scaler = StandardScaler()

# Fit train data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
st.subheader("Hyperparameter Tuning")

# Parameter grid
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

# Create KNN model
knn = KNeighborsClassifier()

# GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

# Train model
grid_search.fit(X_train_scaled, y_train)

# Best model
model = grid_search.best_estimator_

# Save model
joblib.dump(model, "models/knn_classifier.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Show best parameters
st.write("Best Parameters :", grid_search.best_params_)

# Predict output
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Performance section
st.subheader("Model Performance")

# Create columns
c1, c2, c3 = st.columns(3)

# Show metrics
c1.metric("Accuracy", f"{round(accuracy * 100, 2)} %")
c2.metric("Training Rows", X_train.shape[0])
c3.metric("Testing Rows", X_test.shape[0])

# Confusion matrix
st.subheader("Confusion Matrix")

# Generate matrix
cm = confusion_matrix(y_test, y_pred)

# Create figure
fig2, ax2 = plt.subplots(figsize=(5, 5))

# Plot matrix
ax2.imshow(cm)

# Labels
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

# Add values
for i in range(len(cm)):
    for j in range(len(cm)):
        ax2.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=15
        )

# Show graph
st.pyplot(fig2)

# Classification report
st.subheader("Classification Report")

# Generate report
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

# Convert to dataframe
report_df = pd.DataFrame(report).transpose()

# Show report
st.dataframe(report_df)

# Prediction section
st.subheader("Predict House Category")

# Store user input
user_input = {}

# Create columns
col1, col2 = st.columns(2)

# Feature columns
columns = X.columns.tolist()

# Split columns
first_half = columns[:len(columns)//2]
second_half = columns[len(columns)//2:]

# First column inputs
with col1:

    # Loop first half
    for col in first_half:

        # Input field
        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Second column inputs
with col2:

    # Loop second half
    for col in second_half:

        # Input field
        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Predict button
if st.button("Predict Category"):

    # Convert dataframe
    input_df = pd.DataFrame([user_input])

    # Arrange columns
    input_df = input_df[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict result
    prediction = model.predict(input_scaled)

    # Predict probability
    probability = model.predict_proba(input_scaled)

    # Result
    if prediction[0] == 1:

        # High price category
        st.success("High Price House")

    else:

        # Low price category
        st.error("Low Price House")

    # Show probability
    st.info(
        f"High Price Probability : {round(probability[0][1] * 100, 2)} %"
    )