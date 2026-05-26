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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="KNN Regression Project",
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
st.title("House Price Prediction Using KNN Regressor")

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

# Drop unnecessary columns
drop_cols = ["date", "street", "country"]

# Drop columns if present
for col in drop_cols:

    # Check column exists
    if col in df.columns:

        # Drop column
        df = df.drop(col, axis=1)

# Store encoders
encoders = {}

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill missing categorical values
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

# Correlation matrix section
st.subheader("Correlation Matrix")

# Select only numerical columns
numeric_df = df.select_dtypes(include=np.number)

# Create correlation matrix
corr = numeric_df.corr()

# Create figure
fig1, ax1 = plt.subplots(figsize=(12, 8))

# Plot heatmap
heatmap = ax1.imshow(corr)

# Add colorbar
plt.colorbar(heatmap)

# Add axis labels
ax1.set_xticks(range(len(corr.columns)))
ax1.set_yticks(range(len(corr.columns)))

# Column names
ax1.set_xticklabels(corr.columns, rotation=90)
ax1.set_yticklabels(corr.columns)

# Show graph
st.pyplot(fig1)

# Features
X = df.drop("price", axis=1)

# Target
y = df["price"]

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

# Hyperparameter tuning section
st.subheader("Hyperparameter Tuning")

# Parameter grid
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

# Create KNN model
knn = KNeighborsRegressor()

# Grid search
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=3,
    scoring="r2"
)

# Train model
grid_search.fit(X_train_scaled, y_train)

# Best model
model = grid_search.best_estimator_

# Save model
joblib.dump(model, "models/knn_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Show best parameters
st.write("Best Parameters :", grid_search.best_params_)

# Predict output
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Performance section
st.subheader("Model Performance")

# Create columns
c1, c2, c3, c4 = st.columns(4)

# Show metrics
c1.metric("MAE", round(mae, 2))
c2.metric("MSE", round(mse, 2))
c3.metric("RMSE", round(rmse, 2))
c4.metric("R² Score", round(r2, 2))

# Actual vs predicted graph
st.subheader("Actual vs Predicted Prices")

# Create figure
fig2, ax2 = plt.subplots(figsize=(7, 5))

# Scatter plot
ax2.scatter(y_test, y_pred)

# Labels
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")

# Title
ax2.set_title("KNN Regression")

# Show graph
st.pyplot(fig2)

# Prediction section
st.subheader("Predict House Price")

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

        # Numerical input
        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Second column inputs
with col2:

    # Loop second half
    for col in second_half:

        # Numerical input
        user_input[col] = st.number_input(
            f"{col}",
            value=float(df[col].mean())
        )

# Predict button
if st.button("Predict House Price"):

    # Convert into dataframe
    input_df = pd.DataFrame([user_input])

    # Arrange columns correctly
    input_df = input_df[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict price
    prediction = model.predict(input_scaled)

    # Show result
    st.success(
        f"Predicted House Price : ${round(prediction[0], 2)}"
    )