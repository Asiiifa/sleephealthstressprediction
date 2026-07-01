import os
from pathlib import Path
import threading
import webbrowser

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session, send_from_directory, Response
import joblib
import pandas as pd
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def is_debug_mode():
    return os.getenv("FLASK_DEBUG", "0") == "1"


def debug_log(message):
    if is_debug_mode():
        print(f"[DEBUG] {message}")


def mask_secret(value):
    if not value:
        return "missing"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

secret = os.getenv("SECRET_KEY") or "dev-secret-key"
if not os.getenv("SECRET_KEY"):
    debug_log("SECRET_KEY not found. Using development fallback secret.")
app.secret_key = secret

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
debug_log(f".env path: {BASE_DIR / '.env'}")
debug_log(f"SUPABASE_URL loaded: {SUPABASE_URL or 'missing'}")
debug_log(f"SUPABASE_KEY loaded: {mask_secret(SUPABASE_KEY)}")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("[WARN] Supabase is disabled: SUPABASE_URL or SUPABASE_KEY is missing.")

# Load Models
stress_model = joblib.load(BASE_DIR / "ml" / "stress_model.pkl")
sleep_model = joblib.load(BASE_DIR / "ml" / "sleep_model.pkl")


@app.after_request
def disable_cache(response):
    if is_debug_mode():
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def supabase_required_error():
    return jsonify({
        "ok": False,
        "error": "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to your .env file."
    }), 500


def session_user_payload():
    return {
        "user_name": session.get("user_name"),
        "user_email": session.get("user_email")
    }


@app.route("/")
def home():
    debug_log("Rendering template: index.html")
    return render_template("index.html", **session_user_payload())


# ================= AUTH =================
@app.route("/robots.txt")
def robots():
    return send_from_directory(app.static_folder, "robots.txt")


@app.route("/sitemap.xml")
def sitemap():
    pages = [
        {"url": "https://nidra-ai-drgl.onrender.com/", "priority": "1.0"},
    ]
    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_parts.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    for page in pages:
        xml_parts.append(f'<url><loc>{page["url"]}</loc><priority>{page["priority"]}</priority></url>')
    xml_parts.append('</urlset>')
    return Response(''.join(xml_parts), mimetype='application/xml')


@app.route("/auth/session", methods=["POST"])
def update_session():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()

    if not name or not email:
        return jsonify({"ok": False, "error": "Name and email are required"}), 400

    session["user_name"] = name
    session["user_email"] = email
    return jsonify({"ok": True})


@app.route("/auth/signup", methods=["POST"])
def signup():
    if supabase is None:
        return supabase_required_error()

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"ok": False, "error": "All fields required"}), 400

    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        user = res.user
        if user:
            try:
                supabase.table("profiles").insert({
                    "id": user.id,
                    "name": name,
                    "email": email
                }).execute()
            except Exception:
                pass
            session["user_id"] = user.id
            session["user_name"] = name
            session["user_email"] = email
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Signup failed"}), 400
    except Exception as exc:
        print(f"SIGNUP ERROR: {repr(exc)}", flush=True)
        debug_log(f"Signup failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": False, "error": "Something went wrong"}), 500


@app.route("/auth/login", methods=["POST"])
def login():
    if supabase is None:
        return supabase_required_error()

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "All fields required"}), 400

    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        user = res.user
        if user:
            profile = supabase.table("profiles").select("name").eq("id", user.id).execute()
            name = profile.data[0]["name"] if profile.data else email
            session["user_id"] = user.id
            session["user_name"] = name
            session["user_email"] = email
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Invalid credentials"}), 401
    except Exception as exc:
        debug_log(f"Login failed: {exc}")
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        age = int(payload.get("age", 0))
        sleep_duration = float(payload.get("sleep_duration", 0))
        sleep_quality = int(payload.get("sleep_quality", 0))
        daily_steps = int(payload.get("daily_steps", 0))
        activity = int(payload.get("activity", 0))
        resting_hr = int(payload.get("resting_hr", 0))
        current_stress = int(payload.get("current_stress", 0))

        features = pd.DataFrame([{
            "age": age,
            "sleep_duration": sleep_duration,
            "sleep_quality": sleep_quality,
            "daily_steps": daily_steps,
            "activity": activity,
            "resting_hr": resting_hr,
            "current_stress": current_stress
        }])

        stress_value = float(stress_model.predict(features)[0])
        sleep_value = float(sleep_model.predict(features)[0])
        stress_value = max(0, min(100, stress_value))
        sleep_value = max(0, min(100, sleep_value))

        if stress_value > 70:
            level = "High"
        elif stress_value > 40:
            level = "Medium"
        else:
            level = "Low"

        try:
            rag_query = f"My sleep score is {round(sleep_value)}, stress level is {level}, sleep duration is {sleep_duration} hours, sleep quality is {sleep_quality}/10, daily steps are {daily_steps}, activity is {activity} minutes, resting heart rate is {resting_hr} bpm. Give me personalized advice."
            advice = get_rag_response(rag_query)
        except Exception as e:
            print(f"RAG ERROR PREDICT: {repr(e)}", flush=True)
            advice = f"{level} stress detected. Monitor your sleep and activity levels."

        if session.get("user_id") and supabase is not None:
            supabase.table("predictions").insert({
                "user_id": session["user_id"],
                "stress_score": round(stress_value),
                "sleep_score": round(sleep_value),
                "level": level,
                "advice": advice
            }).execute()

        return jsonify({
            "stress_score": round(stress_value),
            "sleep": round(sleep_value),
            "level": level,
            "advice": advice
        })
    except Exception as exc:
        debug_log(f"Prediction failed: {exc}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# ================= CHAT =================
from rag.rag_engine import get_rag_response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message") or ""
    if not msg.strip():
        return jsonify({"reply": "Please ask me something about sleep or stress."})
    try:
        reply = get_rag_response(msg)
    except Exception as e:
        print(f"RAG ERROR CHAT: {repr(e)}", flush=True)
        reply = f"DEBUG ERROR: {repr(e)}"
    return jsonify({"reply": reply})

# ================= HISTORY =================
@app.route("/history")
def history():
    if not session.get("user_id") or supabase is None:
        return jsonify({"labels": [], "stress": [], "sleep": []})

    try:
        res = supabase.table("predictions") \
            .select("stress_score, sleep_score, created_at") \
            .eq("user_id", session["user_id"]) \
            .order("created_at") \
            .execute()

        labels, stress_vals, sleep_vals = [], [], []
        for row in res.data:
            labels.append(row["created_at"][11:19])
            stress_vals.append(row["stress_score"])
            sleep_vals.append(row["sleep_score"])

        return jsonify({"labels": labels, "stress": stress_vals, "sleep": sleep_vals})
    except Exception as exc:
        debug_log(f"History load failed: {exc}")
        return jsonify({"labels": [], "stress": [], "sleep": []})


# ================= RUN =================
if __name__ == "__main__":
    debug = is_debug_mode()
    port = int(os.getenv("PORT", "5000"))
    auto_open = os.getenv("AUTO_OPEN_BROWSER", "1") == "1"
    should_open_browser = auto_open and (
        not debug or os.getenv("WERKZEUG_RUN_MAIN") == "true"
    )

    if should_open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()

    app.run(debug=debug, host="127.0.0.1", port=port)


