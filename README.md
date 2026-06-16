<img width="1331" height="750" alt="image" src="https://github.com/user-attachments/assets/a05f8d4c-aeee-4f80-ae0c-759bfc6d343f" />

NIDRA AI — Sleep Health & Stress Prediction

An AI-powered wellness tool that predicts your stress level and sleep score based on daily lifestyle inputs.

🔗 **Live App:** [https://nidra-ai-drgl.onrender.com](https://nidra-ai-drgl.onrender.com)

---

## 📌 What It Does

- Predicts **Stress Score** (0–100)
- Predicts **Sleep Score** (0–100)
- Classifies stress as Low / Medium / High
- Gives **personalized health suggestions**
- Shows **stress & sleep trend charts** over time
- Includes a basic **chat assistant** for sleep & stress guidance

<img width="1888" height="953" alt="image" src="https://github.com/user-attachments/assets/047179dd-d996-4c34-81ea-a15987a472bb" />
<img width="1897" height="963" alt="image" src="https://github.com/user-attachments/assets/1a3d83c4-e872-48fd-bf6a-4e93b65cc0ea" />

---

## 🧠 Machine Learning

| Detail | Info |
|---|---|
| Algorithm | Random Forest Regressor |
| Library | Scikit-Learn |
| Dataset | [Kaggle — Sleep Health & Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) |
| Model Accuracy | ~90% (R² Score) |

### Input Features Used for Prediction
- Age
- Sleep Duration
- Sleep Quality
- Daily Steps
- Physical Activity Level
- Resting Heart Rate
- Current Stress Level

### How Models Were Trained
1. Kaggle dataset loaded and cleaned
2. Features normalized using StandardScaler
3. Two separate Random Forest models trained — one for stress, one for sleep
4. Models saved as `.pkl` files using Joblib

---

## 🚀 Features

- ✅ Stress & Sleep Score Prediction
- ✅ Real-time trend visualization (Chart.js)
- ✅ Chat assistant for wellness guidance
- ✅ Prediction history saved automatically
- ✅ Dark UI with responsive design
- ✅ Session-based user auth (name + email)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| ML | Scikit-Learn, Pandas, NumPy |
| Deployment | Render (Free Tier) |
| Server | Gunicorn |

---

## 📁 Folder Structure

```
sleephealthstressprediction/
│
├── app.py                  # Flask backend
├── requirements.txt        # Dependencies
├── runtime.txt             # Python version (3.11.0)
│
├── ml/
│   ├── stress_model.pkl    # Trained stress model
│   └── sleep_model.pkl     # Trained sleep model
│
├── templates/
│   └── index.html          # Main UI
│
└── static/
    ├── style.css
    ├── app.js
    ├── logo.png
    └── history.csv         # Auto-saved prediction logs
```

---

## 🖥️ Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Asiiifa/sleephealthstressprediction.git
cd sleephealthstressprediction
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Create `.env` File
```
SECRET_KEY=your_random_secret_key
```

### 5. Run the App
```bash
python app.py
```

### 6. Open in Browser
```
http://127.0.0.1:5000
```

---

## ⚠️ Note on Free Deployment

This app is hosted on Render's free tier.  
The server **spins down after inactivity** — first load may take 30–50 seconds.  
This is expected behavior, not a bug.

---

## 📊 Dataset

**Source:** [Kaggle — Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

The dataset includes lifestyle and health data of individuals including sleep duration, stress levels, physical activity, heart rate, and more.

> Dataset is not included in this repository. Models are pre-trained and included as `.pkl` files.

---

## 👩‍💻 Built By

**Asifa Hamid Khan**  
Academic final year major Project — Sleep Health & Stress Prediction using ML
