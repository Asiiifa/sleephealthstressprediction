import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

os.makedirs("ml", exist_ok=True)

# Use only real data. You can override with:
#   set DATASET_PATH=path\to\your_real_data.csv
DATASET_PATH = os.environ.get("DATASET_PATH", "ml/real_dataset.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Real dataset not found: {DATASET_PATH}\n"
        "Please provide a real CSV file and set DATASET_PATH if needed."
    )

df = pd.read_csv(DATASET_PATH)
df.columns = [c.strip().lower() for c in df.columns]

# Map common real-dataset column names to backend-compatible names.
rename_map = {
    "sleep duration": "sleep_duration",
    "quality of sleep": "sleep_quality",
    "physical activity level": "activity",
    "heart rate": "resting_hr",
    "daily steps": "daily_steps",
    "stress level": "current_stress",
}
for source, target in rename_map.items():
    if source in df.columns and target not in df.columns:
        df[target] = df[source]

# Build score targets from real measured columns if score columns are absent.
if "stress_score" not in df.columns and "current_stress" in df.columns:
    df["stress_score"] = pd.to_numeric(df["current_stress"], errors="coerce") * 10.0

if "sleep_score" not in df.columns and "sleep_quality" in df.columns:
    df["sleep_score"] = pd.to_numeric(df["sleep_quality"], errors="coerce") * 10.0

df = df.drop_duplicates().dropna()

# Must match backend /predict feature order exactly.
feature_columns = [
    "age",
    "sleep_duration",
    "sleep_quality",
    "daily_steps",
    "activity",
    "resting_hr",
    "current_stress",
]
target_stress = "stress_score"
target_sleep = "sleep_score"

required_columns = set(feature_columns + [target_stress, target_sleep])
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

X = df[feature_columns]
y_stress = df[target_stress]
y_sleep = df[target_sleep]

X_train, X_test, y_stress_train, y_stress_test, y_sleep_train, y_sleep_test = train_test_split(
    X, y_stress, y_sleep, test_size=0.2, random_state=42
)

stress_model = RandomForestRegressor(n_estimators=300, random_state=42)
sleep_model = RandomForestRegressor(n_estimators=300, random_state=42)

stress_model.fit(X_train, y_stress_train)
sleep_model.fit(X_train, y_sleep_train)

stress_pred = stress_model.predict(X_test)
sleep_pred = sleep_model.predict(X_test)

print(f"Rows used (real, deduplicated): {len(df)}")
print(f"Stress MAE: {mean_absolute_error(y_stress_test, stress_pred):.2f}")
print(f"Sleep MAE: {mean_absolute_error(y_sleep_test, sleep_pred):.2f}")

joblib.dump(stress_model, "ml/stress_model.pkl")
joblib.dump(sleep_model, "ml/sleep_model.pkl")

print("Models trained on real data and saved to ml/*.pkl")