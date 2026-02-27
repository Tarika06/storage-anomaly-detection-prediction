# 🗄️ AI-Powered Storage Anomaly Detection & Prediction System

An end-to-end machine learning system that monitors storage drive telemetry, detects anomalies, predicts failures before they happen, and explains every decision — built to reflect real-world enterprise data infrastructure challenges.

> Inspired by the kind of intelligent data infrastructure problems solved at companies like NetApp.

---

## 🎯 Project Overview

Enterprise storage systems generate massive amounts of telemetry data every second — IOPS, latency, throughput, temperature, error counts. When something is about to go wrong, the warning signs are hidden in that data. This system surfaces those warning signs early using machine learning.

**What this system does:**
- Ingests real hard drive telemetry data (Backblaze dataset — millions of drive days)
- Engineers time-series features from raw metrics
- Detects anomalous drive behavior using Isolation Forest
- Predicts future metric values and flags drives trending toward failure using Facebook Prophet
- Simulates ransomware detection via mass-write anomaly pattern recognition
- Explains every flagged anomaly using SHAP values
- Visualizes everything in a real-time interactive Streamlit dashboard

---

## 🏗️ Architecture
```
        Raw Telemetry Data (Backblaze)
                     ↓
        Data Ingestion & Cleaning
                     ↓
        Feature Engineering (rolling averages, lag features, rate of change)
                     ↓
┌─────────────────────┬──────────────────────┐
│  Anomaly Detection  │  Failure Prediction  │
│  (Isolation Forest) │  (Facebook Prophet)  │
└─────────────────────┴──────────────────────┘
                     ↓
           SHAP Explainability Layer
                     ↓
            Streamlit Dashboard
```

---

## 🔍 Key Features

**Anomaly Detection**
Unsupervised Isolation Forest model trained on healthy drive behavior. Flags drives deviating significantly from their normal operating baseline with an anomaly score and timestamp.

**Failure Prediction**
Time-series forecasting on key metrics (IOPS, latency, temperature). Raises predictive alerts when projected values are trending toward dangerous thresholds before failure occurs.

**Ransomware Behavior Detection**
Simulates the mass-write signature of ransomware encryption activity and detects it as a critical anomaly, triggering an automated snapshot response log.

**Explainable AI**
Every flagged anomaly comes with a SHAP explanation — showing exactly which metrics contributed to the flag and by how much. Built for trust and transparency.

**Interactive Dashboard**
Real-time Streamlit app showing drive health status, prediction charts, anomaly scores, SHAP panels, and a full alert log.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (Isolation Forest), Facebook Prophet |
| Deep Learning (optional) | PyTorch (LSTM) |
| Explainability | SHAP |
| Visualization | Streamlit, Matplotlib, Seaborn |
| Dataset | Backblaze Hard Drive Stats |

---

## 📁 Project Structure
```
storage-anomaly-detection/
│
├── data/
│   ├── raw/                  # Raw Backblaze CSV files
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_exploration.ipynb  # Data understanding and EDA
│   ├── 02_features.ipynb     # Feature engineering
│   ├── 03_anomaly.ipynb      # Isolation Forest model
│   └── 04_prediction.ipynb   # Prophet forecasting
│
├── models/
│   ├── isolation_forest.pkl  # Saved anomaly detection model
│   └── prophet_model/        # Saved Prophet models per drive
│
├── src/
│   ├── preprocess.py         # Data cleaning functions
│   ├── features.py           # Feature engineering functions
│   ├── anomaly_model.py      # Anomaly detection logic
│   ├── prediction_model.py   # Forecasting logic
│   └── shap_explainer.py     # Explainability functions
│
├── app/
│   └── dashboard.py          # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/storage-anomaly-detection.git
cd storage-anomaly-detection
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Go to [Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) and download one quarter of data. Place the CSV files inside `data/raw/`.

**5. Run the notebooks in order**
```
notebooks/01_exploration.ipynb → 02_features.ipynb → 03_anomaly.ipynb → 04_prediction.ipynb
```

**6. Launch the dashboard**
```bash
streamlit run app/dashboard.py
```

---

## 📊 Dataset

This project uses the **Backblaze Hard Drive Stats** dataset — real telemetry data from hundreds of thousands of drives in Backblaze's data centers, released publicly every quarter since 2013.

Each row represents one drive on one day and includes S.M.A.R.T. attributes like reallocated sector count, read error rate, temperature, power-on hours, and more.

---

## 🔮 Future Improvements

- LSTM-based deep learning model for improved sequence prediction
- Multi-drive correlation analysis (if one drive fails, nearby drives are at higher risk)
- Cloud deployment on AWS or GCP
- Real-time data streaming with Apache Kafka
- Automated retraining pipeline with MLflow tracking

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — feel free to use and build on this project.