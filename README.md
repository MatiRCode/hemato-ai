# HematoAI: Clinical Decision Support System (CDSS)

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Production-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**HematoAI** is an intelligent Clinical Decision Support System designed to assist in the triage and diagnosis of hematological pathologies. Unlike traditional "Black Box" models, this system prioritizes **Clinical Safety** and **Explainability (XAI)**, allowing healthcare professionals to validate the biological logic behind every algorithmic prediction.

---

## Key Features

### 1. Hybrid Inference Engine

- **Classification:** Implements an **XGBoost Classifier** optimized for imbalanced tabular data, capable of distinguishing between Iron Deficiency Anemia, Megaloblastic Anemia, Leukemia, and Infections.
- **Anomaly Detection (OOD):** Integrates an **Isolation Forest** model to detect "Out-of-Distribution" patient profiles that the model has not been trained on (e.g., Polycythemia Vera), preventing hallucinations and reducing false positives.

### 2. Safety Layer Architecture (v2)

The system implements a robust validation layer to ensure clinical safety:

- **Hard Rules (Sanity Checks):** Immediate blocking mechanism for biologically incompatible values (e.g., Hemoglobin < 2.0 g/dL).
- **Uncertainty Logic:**
  - **Valid (Green):** Safe prediction within normal statistical parameters.
  - **Warning (Amber):** Statistical anomaly detected, but with high classification confidence. The system flags the risk but provides the diagnosis.
  - **Block (Red):** Anomaly detected with low classification confidence. The system censors the prediction to prevent potential medical errors.

### 3. Explainability (White-Box AI)

- **SHAP Values (Shapley Additive Explanations):** Real-time visualization of decision drivers. It highlights exactly which feature (HGB, MCV, PLT) influenced the diagnosis.
- **Vector Morphology:** Radar chart visualization to evaluate the patient's hemogram "fingerprint" at a glance.

### 4. Data Persistence

- Local historical record system using SQLite for case tracking and model decision auditing.

---

## Tech Stack

- **Frontend:** Streamlit (Custom CSS for Medical UI).
- **Machine Learning:** XGBoost, Scikit-learn, Isolation Forest.
- **XAI:** SHAP, Matplotlib.
- **Data Engineering:** Pandas, Numpy.
- **Persistence:** SQLite3.

---

## Project Structure

```text
PROYECTO WOW/
├── app.py                 # Entry Point (Main Controller)
├── requirements.txt       # Python Dependencies
├── packages.txt           # System Dependencies (Linux/OpenMP)
├── assets/
│   └── style.css          # Custom Styles (Clinical UX)
├── models/
│   ├── xgboost_clinical_v2.json      # Predictive Model
│   ├── anomaly_detector_v2.joblib    # Anomaly Detector
│   └── label_encoder_v2.joblib       # Class Mapping
└── src/
    └── db_manager.py      # Database Manager
```

---

## Installation and Local Usage

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/matircode/hemato-ai.git](https://github.com/matircode/hemato-ai.git)
    cd hemato-ai
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## Medical Disclaimer

**IMPORTANT NOTICE:**
This software is a functional prototype (MVP) developed for Data Science portfolio and educational demonstration purposes.

- **It is NOT a certified medical device.**
- It must not be used for actual patient diagnosis without supervision.
- Predictions are based on synthetic and public datasets (NHANES) and may not reflect the full complexity of real-world clinical cases.

---

## Author

Developed by **Matias Gacitua Ruiz (MatiRCode)**.
_Clinical Laboratory Scientist._

---
