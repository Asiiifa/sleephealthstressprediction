import os
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
HISTORY_PATH = os.path.join(STATIC_DIR, "history.csv")

app = Flask(__name__, static_folder="static")


def create_and_save_dummy_model(model_path: str) -> None:
    """
    Create a simple synthetic ML model for predicting
    stress level and sleep score, then save it to model_path.
    """
    rng = np.random.RandomState(42)
    n_samples = 500

    ages = rng.randint(18, 65, size=n_samples)
    genders = rng.randint(0, 2, size=n_samples)  # 0 or 1
    sleep_duration = rng.uniform(4, 9, size=n_samples)  # hours
    sleep_quality = rng.randint(1, 6, size=n_samples)   # 1–5 scale
    daily_steps = rng.randint(1000, 15000, size=n_samples)
    heart_rate = rng.randint(55, 100, size=n_samples)

    X = np.column_stack(
        [
            ages,
            genders,
            sleep_duration,
            sleep_quality,
            daily_steps,
            heart_rate,
        ]
    )

    # Synthetic, somewhat realistic relationships
    stress = (
        80
        - 2.0 * (sleep_duration - 6.0)
        - 3.0 * (sleep_quality - 3.0)
        - 0.0008 * daily_steps
        + 0.7 * (heart_rate - 70.0)
        + rng.normal(0, 5, size=n_samples)
    )
    stress = np.clip(stress, 0, 100)

    sleep_score = (
        20.0
        + 8.0 * (sleep_duration - 4.0)
        + 5.0 * (sleep_quality - 1.0)
        + 0.0005 * daily_steps
        - 0.5 * (heart_rate - 65.0)
        - 0.3 * (stress - 50.0)
        + rng.normal(0, 5, size=n_samples)
    )
    sleep_score = np.clip(sleep_score, 0, 100)

    y = np.column_stack([stress, sleep_score])

    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X, y)

    joblib.dump(model, model_path)


def ensure_directories_and_files() -> None:
    """
    Make sure models/ and static/ exist,
    ensure model.pkl and history.csv are present.
    """
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Ensure history.csv has a header
    if not os.path.exists(HISTORY_PATH):
        columns = [
            "timestamp",
            "age",
            "gender",
            "sleep_duration",
            "sleep_quality",
            "daily_steps",
            "heart_rate",
            "predicted_stress",
            "sleep_score",
        ]
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(HISTORY_PATH, index=False)

    # Ensure model.pkl exists
    if not os.path.exists(MODEL_PATH):
        create_and_save_dummy_model(MODEL_PATH)


def load_model():
    """
    Ensure model exists, then load it from disk.
    """
    ensure_directories_and_files()
    loaded_model = joblib.load(MODEL_PATH)
    return loaded_model


# Load or create the model at startup
model = load_model()


@app.route("/")
def index():
    """
    Serve the main HTML page.
    """
    return send_from_directory(STATIC_DIR, "index.html")


def preprocess_input(data: dict) -> dict:
    """
    Validate and preprocess incoming JSON payload.
    Returns a dict with cleaned values and numpy features array.
    """
    required_fields = [
        "age",
        "gender",
        "sleep_duration",
        "sleep_quality",
        "daily_steps",
        "heart_rate",
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    try:
        age = float(data["age"])
        sleep_duration = float(data["sleep_duration"])
        sleep_quality = float(data["sleep_quality"])
        daily_steps = float(data["daily_steps"])
        heart_rate = float(data["heart_rate"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric fields must be valid numbers.") from exc

    gender_raw = str(data["gender"]).strip().lower()
    if gender_raw in ("male", "m"):
        gender_encoded = 1.0
    elif gender_raw in ("female", "f"):
        gender_encoded = 0.0
    else:
        # For other / unspecified genders, use a mid value
        gender_encoded = 0.5

    features = np.array(
        [
            [
                age,
                gender_encoded,
                sleep_duration,
                sleep_quality,
                daily_steps,
                heart_rate,
            ]
        ],
        dtype=float,
    )

    return {
        "age": age,
        "gender_raw": gender_raw,
        "gender_encoded": gender_encoded,
        "sleep_duration": sleep_duration,
        "sleep_quality": sleep_quality,
        "daily_steps": daily_steps,
        "heart_rate": heart_rate,
        "features": features,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Receive JSON, run prediction, append to history.csv, return JSON result.
    """
    if not request.is_json:
        return (
            jsonify(
                {"error": "Request content-type must be application/json"}
            ),
            400,
        )

    data = request.get_json()

    try:
        processed = preprocess_input(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    predictions = model.predict(processed["features"])
    predicted_stress = float(predictions[0][0])
    sleep_score = float(predictions[0][1])

    predicted_stress_rounded = round(predicted_stress, 2)
    sleep_score_rounded = round(sleep_score, 2)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "age": processed["age"],
        "gender": processed["gender_raw"],
        "sleep_duration": processed["sleep_duration"],
        "sleep_quality": processed["sleep_quality"],
        "daily_steps": processed["daily_steps"],
        "heart_rate": processed["heart_rate"],
        "predicted_stress": predicted_stress_rounded,
        "sleep_score": sleep_score_rounded,
    }

    # Safely append to history.csv
    try:
        df = pd.DataFrame([record])

        if not os.path.exists(HISTORY_PATH):
            # If somehow deleted, recreate with header
            df.to_csv(HISTORY_PATH, index=False)
        else:
            df.to_csv(
                HISTORY_PATH,
                mode="a",
                header=False,
                index=False,
            )
    except Exception as exc:
        # In a real app you might log this properly
        print(f"Failed to append to history.csv: {exc}")

    return jsonify(
        {
            "predicted_stress": predicted_stress_rounded,
            "sleep_score": sleep_score_rounded,
        }
    )


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Return the prediction history as JSON.
    """
    if not os.path.exists(HISTORY_PATH):
        return jsonify([])

    try:
        df = pd.read_csv(HISTORY_PATH)
        history = df.to_dict(orient="records")
        return jsonify(history)
    except Exception as exc:
        return jsonify({"error": f"Failed to read history: {exc}"}), 500


if __name__ == "__main__":
    # For local development
    app.run(debug=True)
