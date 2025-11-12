import pandas as pd
import joblib
import os
import json # Import the json library

# Imports for the LATEST stable Evidently version (0.4.x)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping

# --- Configuration ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script's directory
REFERENCE_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'reference_data.csv')
CURRENT_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'current_data.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'model.pkl')

# --- NEW: Define paths for multiple report formats and status output ---
HTML_REPORT_PATH = os.path.join(SCRIPT_DIR, '..', 'reports', 'model_performance_dashboard.html')
JSON_REPORT_PATH = os.path.join(SCRIPT_DIR, '..', 'reports', 'model_performance_dashboard.json')
DRIFT_STATUS_PATH = os.path.join(SCRIPT_DIR, '..', 'drift_status.txt')


TARGET_COLUMN = 'cnt'
NUMERICAL_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']
CATEGORICAL_FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print("--- Starting Monitoring Script (Latest API) ---")

# --- 1. Load Data and Model ---
print("Loading data and model...")
reference_data = pd.read_csv(REFERENCE_DATA_PATH)
current_data = pd.read_csv(CURRENT_DATA_PATH)
model = joblib.load(MODEL_PATH)
print("Successfully loaded data and model.")

# --- 2. Generate Predictions ---
print("Generating predictions...")
reference_data['prediction'] = model.predict(reference_data[MODEL_FEATURES])
current_data['prediction'] = model.predict(current_data[MODEL_FEATURES])
print("Predictions generated.")

# --- 3. Define Column Mapping ---
column_mapping = ColumnMapping(
    target=TARGET_COLUMN,
    prediction='prediction',
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES
)

# --- 4. Create and Run the Report ---
print("Creating and running the report...")
report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset(),
])
report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# --- 5. Save the Report in both HTML and JSON formats ---
print(f"Saving HTML report to: {HTML_REPORT_PATH}")
report.save_html(HTML_REPORT_PATH)

print(f"Saving JSON report to: {JSON_REPORT_PATH}")
report.save_json(JSON_REPORT_PATH)

# --- 6. NEW: Check JSON Report for Data Drift ---
print("Checking report for data drift...")
drift_detected = False
with open(JSON_REPORT_PATH, 'r') as f:
    report_json = json.load(f)

# Navigate the JSON structure to find the overall drift status
if report_json['metrics'][0]['result']['dataset_drift']:
    drift_detected = True
    print("DRIFT DETECTED")
else:
    print("No drift detected")

# --- 7. NEW: Create an output file for the workflow ---
# This file will communicate the drift status to the GitHub Action workflow
with open(DRIFT_STATUS_PATH, 'w') as f:
    f.write(str(drift_detected).lower())

print(f"Drift status '{str(drift_detected).lower()}' saved to {DRIFT_STATUS_PATH}")
print("--- Monitoring Script Finished ---")