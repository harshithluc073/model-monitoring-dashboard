import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Configuration ---
# Build paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'hour.csv')
OLD_MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'model.pkl')
NEW_MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'model.pkl') # We will overwrite the old model
VALIDATION_RESULTS_PATH = os.path.join(SCRIPT_DIR, '..', 'validation_results.json')

# Define model features and target
TARGET_COLUMN = 'cnt'
NUMERICAL_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']
CATEGORICAL_FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print("--- Starting Model Retraining and Validation Script ---")

# --- 1. Load Full Dataset ---
print(f"Loading raw data from: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH, parse_dates=['dteday'])
# For this example, we'll use the whole dataset. In a real scenario,
# you might pull the latest data from a database.

# Create a simple train-test split from the full dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_test = test_df[MODEL_FEATURES]
y_test = test_df[TARGET_COLUMN]

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# --- 2. Train a New Model ---
print("Training a new RandomForestRegressor model...")
new_model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train = train_df[MODEL_FEATURES]
y_train = train_df[TARGET_COLUMN]
new_model.fit(X_train, y_train)
print("New model training complete.")

# --- 3. Load the Old Model ---
print(f"Loading old model from: {OLD_MODEL_PATH}")
try:
    old_model = joblib.load(OLD_MODEL_PATH)
    print("Old model loaded successfully.")
except FileNotFoundError:
    print("No old model found. Proceeding with the new model as the first model.")
    old_model = None # Handle the case where no model exists yet

# --- 4. Evaluate Both Models ---
print("Evaluating models on the test set...")
new_predictions = new_model.predict(X_test)
new_rmse = mean_squared_error(y_test, new_predictions, squared=False)
print(f"New Model RMSE: {new_rmse}")

old_rmse = float('inf') # Set old model error to infinity if it doesn't exist
if old_model:
    old_predictions = old_model.predict(X_test)
    old_rmse = mean_squared_error(y_test, old_predictions, squared=False)
    print(f"Old Model RMSE: {old_rmse}")

# --- 5. Compare Performance and Save Results ---
results = {
    "old_model_rmse": old_rmse,
    "new_model_rmse": new_rmse,
    "new_model_is_better": bool(new_rmse < old_rmse)
}

print("Saving validation results...")
with open(VALIDATION_RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Validation results: {results}")

# --- 6. Overwrite Old Model if New One is Better ---
if results["new_model_is_better"]:
    print(f"New model is better. Overwriting old model at: {NEW_MODEL_PATH}")
    joblib.dump(new_model, NEW_MODEL_PATH)
else:
    print("Old model is better or performance is the same. Model not updated.")

print("--- Script Finished Successfully ---")