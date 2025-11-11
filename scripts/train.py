import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- Configuration ---
# Define file paths
RAW_DATA_PATH = os.path.join('..', 'data', 'hour.csv')
REFERENCE_DATA_PATH = os.path.join('..', 'data', 'reference_data.csv')
CURRENT_DATA_PATH = os.path.join('..', 'data', 'current_data.csv')
MODEL_PATH = os.path.join('..', 'models', 'model.pkl')

# Define model features and target
TARGET_COLUMN = 'cnt'
NUMERICAL_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']
CATEGORICAL_FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print("--- Starting Model Training Script ---")

# --- 1. Load and Prepare Data ---
print(f"Loading raw data from: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH, parse_dates=['dteday'])

# Split data to simulate a time-based production scenario
# The first 20 days of the last month will be our "reference" data (what we train on)
# The next 10 days will be our "current" production data (what we monitor)
df_sorted = df.sort_values(by='dteday')
last_month = df_sorted['mnth'].max()
last_year = df_sorted['yr'].max()

# Isolate the last month's data
production_simulation_data = df_sorted[
    (df_sorted['yr'] == last_year) & (df_sorted['mnth'] == last_month)
]

# Split the last month's data
reference_df = production_simulation_data[production_simulation_data['dteday'].dt.day <= 20]
current_df = production_simulation_data[production_simulation_data['dteday'].dt.day > 20]

print(f"Reference data shape: {reference_df.shape}")
print(f"Current data shape: {current_df.shape}")

# Save these datasets for our monitoring pipeline
print(f"Saving reference data to: {REFERENCE_DATA_PATH}")
reference_df.to_csv(REFERENCE_DATA_PATH, index=False)
print(f"Saving current data to: {CURRENT_DATA_PATH}")
current_df.to_csv(CURRENT_DATA_PATH, index=False)

# --- 2. Train the Model ---
print("Training RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)

# We train the model ONLY on the reference data
X_train = reference_df[MODEL_FEATURES]
y_train = reference_df[TARGET_COLUMN]

model.fit(X_train, y_train)
print("Model training complete.")

# --- 3. Save the Model ---
print(f"Saving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print("--- Script Finished Successfully ---")