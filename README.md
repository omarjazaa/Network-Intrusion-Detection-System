# 🛡️ AI-Powered Network Intrusion Detection System (NIDS)

> **An Intelligent Machine Learning & Cybersecurity Solution for Real-Time Network Threat Analysis**

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-111111?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Project_Status-MVP_Release-orange?style=for-the-badge)

---

## 📌 Overview

**Network Intrusion Detection System (NIDS)** is a modern cybersecurity application leveraging Artificial Intelligence and Machine Learning to automatically inspect network traffic, detect anomalies, and classify cyber threats in real time.

This repository represents the **MVP (Minimum Viable Product)** version of the system. It offers an end-to-end operational pipeline: from network traffic simulation and preprocessing to model inference and interactive security monitoring through a dark-themed cybersecurity dashboard.

---

## 📸 Dashboard Preview & Screenshots

### 1️⃣ Main Dashboard Overview
*Real-time threat detection metrics, active security alerts, and dataset execution stats.*
![Dashboard Overview](assets/dashboard_overview.png)

---

### 2️⃣ Threat Visualizations & Analytics
*Interactive Plotly donuts showing attack category distributions alongside the top model features influencing threat detection.*

| Network Traffic Distribution | Top Influencing Features |
| :---: | :---: |
| ![Traffic Distribution](assets/traffic_distribution.png) | ![Feature Importance](assets/feature_importance.png) |

---

### 3️⃣ Detailed Data Logs & Model Specs
*Data inspection logs with CSV export capabilities alongside active model runtime details.*

| Detailed Inspection Logs | Active AI Model Specs |
| :---: | :---: |
| ![Detailed Logs](assets/detailed_logs.png) | ![Model Details](assets/model_details.png) |

---

## 🎯 MVP Key Highlights

* **Dynamic Traffic Generation:** Built-in simulation tool (`generator.py`) capable of generating unique, scenario-driven network traffic samples (varying row sizes, distinct attack ratios, and missing values).
* **Instant One-Click Demo:** Interactive Streamlit dashboard configured to automatically detect and load the latest generated sample with a single click.
* **Dual AI Engine:** Choice between high-accuracy **Random Forest** and high-speed **XGBoost** models.
* **Cybersecurity Dashboard:** Real-time threat metrics, attack distribution donuts, feature importance charts, and downloadable inspection reports (CSV).

---

## 🚨 Detectable Attack Categories

The model classifies incoming connection records into the following categories:

| Traffic Category | Description | Threat Level |
| :--- | :--- | :--- |
| 🟩 **Normal** | Safe, standard network communications | `SAFE` |
| 🟥 **DoS (Denial of Service)** | Flooding attacks aiming to exhaust server resources | `HIGH` |
| 🟧 **PortScan** | Network reconnaissance and vulnerability probes | `MEDIUM` |
| 🟨 **BruteForce** | Automated credential stuffing / password guessing attacks | `HIGH` |
| 🟪 **Botnet** | Command-and-control bot network activities | `CRITICAL` |

---

## 🏗️ Clean Project Structure

The repository follows standard Software Engineering and Data Science clean-code conventions:

```text
Network-Intrusion-Detection-System/
├── assets/                        # Dashboard screenshots & media assets
│   ├── dashboard_overview.png
│   ├── traffic_distribution.png
│   ├── feature_importance.png
│   ├── detailed_logs.png
│   └── model_details.png
├── dashboard/
│   └── app.py                     # Main Streamlit web dashboard application
├── data/
│   └── sample/
│       ├── generator.py           # Dynamic network traffic sample generator script
│       └── sample_network_traffic.csv
├── models/                        # Pre-trained ML model artifacts (.pkl)
├── notebooks/
│   └── experiments.ipynb          # Jupyter Notebook detailing training & evaluation
├── src/
│   ├── data/
│   │   ├── loader.py              # Data ingestion module
│   │   └── preprocessing.py       # Cleaning, encoding, and missing value handling
│   ├── features/
│   │   └── feature_engineering.py # Feature extraction and selection
│   ├── model/
│   │   ├── train.py               # Model training pipeline
│   │   ├── predict.py             # Inference pipeline
│   │   └── evaluate.py            # Classification metrics & evaluation
│   └── visualization/
│       └── plots.py               # Plotly interactive visualization charts
├── .gitignore                     # Git exclusion rules
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
└── LICENSE                        # MIT License
🛠️ Tech Stack & Libraries
Language: Python 3.10+

Machine Learning: scikit-learn, xgboost, pandas, numpy

Visualization: plotly, matplotlib, seaborn

Web Interface: streamlit

Model Persistence: joblib

⚙️ Quickstart & Local Setup
1️⃣ Clone the Repository & Setup Environment
Bash
git clone [https://github.com/OmarAljazaa/Network-Intrusion-Detection-System.git](https://github.com/OmarAljazaa/Network-Intrusion-Detection-System.git)
cd Network-Intrusion-Detection-System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2️⃣ Generate Synthetic Test Samples (Optional)
Generate custom dynamic CSV files for testing:

Bash
# Generate 3 distinct traffic samples
python data/sample/generator.py 3
3️⃣ Run the Streamlit Dashboard
Bash
streamlit run dashboard/app.py
Open your browser at http://localhost:8501.

🧪 Model Experiments & Training
All training steps, hyperparameter tuning, and metric evaluation reports are documented in notebooks/experiments.ipynb.

To re-train the models programmatically:

Bash
python src/model/train.py
🔮 Future Roadmap (Post-MVP Development)
While this MVP validates the core detection and user interface pipeline, future iterations will include:

[ ] Real-Time Packet Sniffing: Integration with Scapy / PyShark for live PCAP packet capture and analysis.

[ ] Deep Learning Models: Implementing LSTM and Autoencoders for zero-day threat detection.

[ ] Instant Alerting: Webhook notification integration (Telegram / Slack / Email) when threat ratio exceeds safety thresholds.

[ ] REST API Service: FastAPI backend wrapper for external SIEM integration.

[ ] Explainable AI (XAI): SHAP / LIME implementation to provide deep insights into individual threat classifications.

👨‍💻 Author
Omar Al-Jazaa

Informatics Engineering Student | Backend & security Specialist

GitHub: @omarjazaa

Specialization: Backend Development, Applied AI in Cybersecurity

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.