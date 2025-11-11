import pandas as pd
import joblib
import os

# Imports for the LATEST stable Evidently version (0.4.x)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping


# --- Configuration ---
REFERENCE_DATA_PATH = os.path.join('..', 'data', 'reference_data.csv')
CURRENT_DATA_PATH = os.path.join('..', 'data', 'current_data.csv')
MODEL_PATH = os.path.join('..', 'models', 'model.pkl')
REPORT_PATH = os.path.join('..', 'reports', 'model_performance_dashboard.html')

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
# The ColumnMapping object is still the correct way to define the data schema
column_mapping = ColumnMapping(
    target=TARGET_COLUMN,
    prediction='prediction',
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES
)

# --- 4. Create and Run the Report ---
print("Creating and running the report...")

# The modern API uses a Report object with a list of metrics (presets are a convenient type of metric)
report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset(),
])

# The .run() method takes the datasets and the column mapping
report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# --- 5. Save the Report ---
# The modern Report object has the .save_html() method directly
report.save_html(REPORT_PATH)

print(f"--- Monitoring Script Finished. Report saved to: {REPORT_PATH} ---")