import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(training_data):
    """Train the regression model and calculate R-squared and RMSE."""
    # Splitting training data into features (X) and target (y)
    numcols = training_data[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = training_data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]

    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])

    bikedf_final = pd.concat([numcols, objcols_dummy], axis=1)

    y = bikedf_final['cnt']
    X = bikedf_final.drop('cnt', axis=1)
    
    # Build model after removing multicollinear columns
    X_new = X.drop(['atemp', 'registered'], axis=1)
    model = LinearRegression().fit(X_new, y)

    # Predictions and metrics
    y_pred = model.predict(X_new)
    r_squared = model.score(X_new, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return model, r_squared, rmse, X_new

def evaluate_model(model, test_data):
    """Evaluate the trained model on test data and calculate RMSE."""
    numcols = test_data[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    objcols = test_data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']]

    objcols_dummy = pd.get_dummies(objcols, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])

    testdf_final = pd.concat([numcols, objcols_dummy], axis=1)

    y_test = testdf_final['cnt']
    X_test = testdf_final.drop('cnt', axis=1)
    
    # Align columns with training data (X_new)
    X_test_aligned = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predictions and metrics
    y_test_pred = model.predict(X_test_aligned)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return rmse_test

# Streamlit App
st.title("Bike Sharing Demand Prediction")

# Upload training data
st.header("Step 1: Upload Training Data")
training_file = st.file_uploader("Upload the training CSV file", type=['csv'])

if training_file is not None:
    training_data = pd.read_csv(training_file)
    st.write("Training Data Preview:", training_data.head())

    # Train the model
    model, r_squared, rmse, X_new = train_model(training_data)
    
    st.subheader("Training Results")
    st.write(f"R² (R-squared): {r_squared:.4f}")
    st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

    # Upload test data
    st.header("Step 2: Upload Test Data")
    test_file = st.file_uploader("Upload the test CSV file", type=['csv'])

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data Preview:", test_data.head())

        # Evaluate the model
        rmse_test = evaluate_model(model, test_data)

        st.subheader("Test Results")
        st.write(f"RMSE on Test Data: {rmse_test:.4f}")
