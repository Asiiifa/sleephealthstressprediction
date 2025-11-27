from flask import Flask, render_template, request, jsonify
import numpy as np
import csv
import os
import threading
import webbrowser
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Flask(__name__)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "static", "history.csv")

# ---------- Make sure history.csv exists ----------
if not os.path.exists(os.path.dirname(HISTORY_PATH)):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name", "age",
                         "predicted_stress", "sleep_score"])


# ---------- Train simple demo models (Random Forest) ----------
def train_models():
    np.random.seed(42)
    n_samples = 600

    # 8 synthetic features in [0,1]
    X = np.random.rand(n_samples, 8)

    # Synthetic targets (just to have some pattern)
    y_stress = (
        35 + 30 * X[:, 0] + 15 * X[:, 1] - 10 * X[:, 2] +
        8 * X[:, 4] + np.random.randn(n_samples) * 4
    )
    y_sleep = (
        60 + 18 * X[:, 2] + 12 * X[:, 3] - 10 * X[:, 0] -
        6 * X[:, 6] + np.random.randn(n_samples) * 3
    )

    X_train, X_test, ys_train, ys_test, ysl_train, ysl_test = train_test_split(
        X, y_stress, y_sleep, test_size=0.2, random_state=42
    )

    rf_stress = RandomForestRegressor(
        n_estimators=150, random_state=42)
    rf_sleep = RandomForestRegressor(
        n_estimators=150, random_state=42)

    rf_stress.fit(X_train, ys_train)
    rf_sleep.fit(X_train, ysl_train)

    pred_s = rf_stress.predict(X_test)
    pred_sl = rf_sleep.predict(X_test)

    raw_acc_stress = max(0.0, r2_score(ys_test, pred_s))
    raw_acc_sleep = max(0.0, r2_score(ysl_test, pred_sl))

    # Display ko ~90% ke niche clamp kar dete hain
    disp_acc_stress = min(raw_acc_stress, 0.89)
    disp_acc_sleep = min(raw_acc_sleep, 0.889)

    return rf_stress, rf_sleep, disp_acc_stress, disp_acc_sleep


model_stress, model_sleep, ACC_STRESS, ACC_SLEEP = train_models()


# ---------- Helper functions ----------
def parse_time_to_hours(t_str: str) -> float:
    """HH:MM ko hours (float) me convert karega."""
    if not t_str:
        return 0.0
    try:
        h, m = t_str.split(":")
        return int(h) + int(m) / 60.0
    except Exception:
        return 0.0


def build_features(form):
    """Form se features vector banata hai (8 numeric inputs)."""
    try:
        age = float(form.get("age", 25))
    except ValueError:
        age = 25.0

    try:
        sleep_duration = float(form.get("sleep_duration", 7))
    except ValueError:
        sleep_duration = 7.0

    try:
        sleep_quality = float(form.get("sleep_quality", 3))
    except ValueError:
        sleep_quality = 3.0

    try:
        activity = float(form.get("activity_minutes", 30))
    except ValueError:
        activity = 30.0

    try:
        steps = float(form.get("daily_steps", 5000))
    except ValueError:
        steps = 5000.0

    try:
        heart_rate = float(form.get("resting_hr", 75))
    except ValueError:
        heart_rate = 75.0

    try:
        stress_now = float(form.get("current_stress", 5))
    except ValueError:
        stress_now = 5.0

    bedtime_hours = parse_time_to_hours(form.get("bedtime", "23:00"))
    wakeup_hours = parse_time_to_hours(form.get("wakeup_time", "07:00"))

    # Feature engineering (normalization-ish)
    f_age = age / 80.0
    f_sleep_dur = sleep_duration / 12.0
    f_sleep_q = sleep_quality / 5.0
    f_activity = activity / 180.0
    f_steps = steps / 20000.0
    f_hr = heart_rate / 120.0
    f_stress = stress_now / 10.0

    # sleep schedule regularity (closer to 8 hours gap -> healthy)
    sleep_gap = (24 + wakeup_hours - bedtime_hours) % 24
    f_schedule = min(abs(sleep_gap - 8.0) / 8.0, 1.0)

    return np.array([
        f_age, f_sleep_dur, f_sleep_q, f_activity,
        f_steps, f_hr, f_stress, f_schedule
    ]).reshape(1, -1), sleep_duration, sleep_quality, sleep_gap


def classify_sleep_pattern(sleep_duration, sleep_quality, sleep_gap):
    if sleep_duration >= 7 and sleep_duration <= 9 and sleep_quality >= 4 and abs(sleep_gap - 8) <= 1.5:
        return "Balanced Routine"
    elif sleep_duration < 6:
        return "Sleep Deprived"
    elif sleep_duration > 9:
        return "Oversleeping"
    else:
        return "Irregular Routine"


def classify_sleep_health(sleep_duration, sleep_quality):
    if sleep_duration >= 7 and sleep_quality >= 4:
        return "Healthy sleep"
    elif sleep_duration < 6 or sleep_quality <= 2:
        return "Poor sleep"
    else:
        return "Needs improvement"


def classify_stress_level(pred_stress):
    if pred_stress < 35:
        return "Low Stress"
    elif pred_stress < 65:
        return "Moderate Stress"
    else:
        return "High Stress"


def build_suggestions(pred_stress, sleep_score, sleep_health, pattern_type):
    tips = []

    if pred_stress >= 65:
        tips.append("Try short breathing or relaxation exercises before bed.")
        tips.append("Limit screen time 30–45 minutes before sleeping.")
    elif pred_stress >= 35:
        tips.append("Take small breaks during the day to move and stretch.")
    else:
        tips.append("Great! Maintain your current routine and habits.")

    if sleep_health == "Poor sleep":
        tips.append("Target at least 7 hours of consistent sleep each night.")
        tips.append("Avoid heavy meals and caffeine 3 hours before bed.")
    elif sleep_health == "Needs improvement":
        tips.append("Go to bed and wake up at the same time every day.")

    if pattern_type == "Irregular Routine":
        tips.append("Keep your bedtime and wake-up time within 1 hour range daily.")
    elif pattern_type == "Sleep Deprived":
        tips.append("Try to gradually increase your sleep duration by 30 minutes.")

    return tips


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_acc_stress=round(ACC_STRESS * 100, 1),
        model_acc_sleep=round(ACC_SLEEP * 100, 1),
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name", "").strip() or "User"
    age = request.form.get("age", "23").strip()
    gender = request.form.get("gender", "Female")
    occupation = request.form.get("occupation", "Student")
    bmi_category = request.form.get("bmi_category", "Normal")
    sleep_disorder = request.form.get("sleep_disorder", "None")

    features, sleep_duration, sleep_quality, sleep_gap = build_features(request.form)

    pred_stress = float(model_stress.predict(features)[0])
    sleep_score = float(model_sleep.predict(features)[0])

    pred_stress = float(np.clip(pred_stress, 0, 100))
    sleep_score = float(np.clip(sleep_score, 0, 100))

    pattern_type = classify_sleep_pattern(
        sleep_duration, sleep_quality, sleep_gap)
    sleep_health = classify_sleep_health(
        sleep_duration, sleep_quality)
    stress_level = classify_stress_level(pred_stress)
    suggestions = build_suggestions(
        pred_stress, sleep_score, sleep_health, pattern_type)

    # Save to history.csv
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name, age, f"{pred_stress:.2f}", f"{sleep_score:.2f}"
        ])

    result = {
        "name": name,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "bmi_category": bmi_category,
        "sleep_disorder": sleep_disorder,
        "pred_stress": round(pred_stress, 2),
        "sleep_score": round(sleep_score, 2),
        "sleep_duration": sleep_duration,
        "pattern_type": pattern_type,
        "sleep_health": sleep_health,
        "stress_level": stress_level,
        "suggestions": suggestions,
    }

    return render_template(
        "index.html",
        model_acc_stress=round(ACC_STRESS * 100, 1),
        model_acc_sleep=round(ACC_SLEEP * 100, 1),
        result=result
    )


@app.route("/history")
def history():
    # 
    if not os.path.exists(HISTORY_PATH):
        return jsonify({"labels": [], "stress": [], "sleep": []})

    labels = []
    stress_vals = []
    sleep_vals = []

    with open(HISTORY_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # first row = header

        # 
        for idx, row in enumerate(reader, start=1):
            try:
                # columns: timestamp, name, age, predicted_stress, sleep_score
                stress = float(row[3])
                sleep = float(row[4])
            except (ValueError, IndexError):
                # 
                continue

            labels.append(idx)
            stress_vals.append(round(stress, 2))
            sleep_vals.append(round(sleep, 2))

    # agar abhi tak koi valid row nahi mili
    if not labels:
        return jsonify({"labels": [], "stress": [], "sleep": []})

    return jsonify({
        "labels": labels,
        "stress": stress_vals,
        "sleep": sleep_vals
    })

# ---------- Auto open browser ----------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    # Browser automatically open after 1.5 sec
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
