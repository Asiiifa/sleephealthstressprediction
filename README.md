# 🌙 Sleep Health & Stress Prediction  
### An AI-powered wellness analysis project using Machine Learning & Flask  

This project predicts a user's **stress level**, **sleep score**, **sleep health**, and gives **personalized suggestions** based on their daily lifestyle inputs.  
It also visualizes **stress trends and sleep patterns** over time.

Built as part of an academic project, aligned fully with the project PPT structure.

---

## 🚀 Features  

### 🔹 **1. Stress & Sleep Prediction**
- Predicts **stress level** (0–100)
- Predicts **sleep score** (0–100)
- Classifies **Sleep Pattern** (Balanced Routine, Sleep Deprivation, etc.)
- Evaluates **Overall Sleep Health**

### 🔹 **2. Advanced Inputs Covered**
- Age  
- Gender  
- Occupation  
- BMI Category  
- Sleep Duration  
- Bedtime & Wake-up Time  
- Daily Steps  
- Physical Activity  
- Resting Heart Rate  
- Current Stress Level  
- Sleep Disorders  

(Exactly as shown in the PPT)

### 🔹 **3. Data Visualization**
- Stress Trend Chart  
- Sleep Score Trend Chart  
- Automatic history saving in `history.csv`  
- Shows long-term behavior  

### 🔹 **4. Machine Learning**
- Random Forest Algorithm  
- ~90% accuracy (as described in the PPT)  
- Data Preprocessing:  
  - Missing value handling  
  - Normalization  
  - Encoding  
  - Correlation analysis  

### 🔹 **5. Modern Web UI**
- Dark/Light Theme  
- Responsive design  
- Smooth UI transitions  
- Fully modernized HTML + CSS + JS  
- Auto-load interface  

### 🔹 **6. Auto-generated Insights**
- Personalized suggestions  
- Sleep health interpretation  
- Balanced routine detection  

---

## 🧠 Machine Learning Model

### 📌 Algorithm Used
✔ Random Forest Classifier  
✔ Selected because:
- High accuracy  
- Handles mixed data  
- Good for classification tasks  

### 📌 Model Accuracy
As per analysis and PPT slides:  
- **Stress Model Accuracy:** ~90%  
- **Sleep Score Accuracy:** ~90%

---

## 📊 Dataset Details
Used dataset: **Kaggle Sleep Health & Lifestyle Dataset**

Includes data fields like:
- Gender
- Age
- Occupation
- Stress
- Sleep Duration
- Sleep Quality
- Heart Rate  
… and more.

---
## 📁 Folder Structure 

sleephealthstressprediction/
│
├── app.py # Main Flask backend (Python)
├── models/
│ └── model.pkl # Trained ML model
│
├── templates/
│ └── index.html # Frontend UI template
│
├── static/
│ ├── style.css # Main CSS styling
│ ├── script.js # All JavaScript functions
│ └── history.csv # Auto-saved predictions
│
└── requirements.txt # Libraries used

markdown
Copy code

### 📌 Why these folders?
- **templates/** → Flask loads HTML only from this folder  
- **static/** → All CSS, JS, images stored here  
- **models/** → ML model files  
- **history.csv** → Stores prediction logs  

---

## 🛠️ Tech Stack

### **Frontend**
- HTML  
- CSS  
- JavaScript  
- Responsive UI  

### **Backend**
- Python  
- Flask  

### **Machine Learning**
- Scikit-Learn  
- Pandas  
- NumPy  

---

## 🛠 Installation & Running the App

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Asiiifa/sleephealthstressprediction.git

2️⃣ Create Virtual Environment
python -m venv venv

3️⃣ Activate Virtual Environment

Windows:

venv\Scripts\activate

4️⃣ Install Requirements
pip install -r requirements.txt

5️⃣ Run the App
python app.py

6️⃣ Open in Browser
http://127.0.0.1:5000


---

