# 🏦 Two-Stage Loan Underwriting & Risk Prediction FastAPI Service
### B.Tech 2nd Year CSE Machine Learning Project

A production-ready **FastAPI** web service for loan eligibility and repayment risk scoring. Ready to deploy on **Render** (free cloud web service) or run inside **Kaggle Notebooks**.

---

## 📌 Project Overview & Decision Logic

In commercial banking, loan decisions follow a sequential two-stage evaluation:

1. **Stage 1 (Loan Approval / Eligibility):**
   - Evaluates whether the applicant qualifies for a loan based on credit score, income, and debt-to-income (DTI) ratio.
   - **Target:** `1` = Approved, `0` = Rejected.
   - **Accuracy:** > 98% | **ROC-AUC:** > 99%.
   - **Gatekeeper:** If rejected, the application is declined immediately.

2. **Stage 2 (Repayment Default Risk):**
   - Only evaluated for **Approved** loans (Zero Target Leakage).
   - Predicts whether the borrower is likely to pay back on time or default.
   - **Target:** `1` = Defaulted, `0` = Repaid.
   - **Accuracy:** > 85% | **ROC-AUC:** > 93%.
   - If default risk $\ge 40\%$, flags as **Conditional Approval** (requires co-signer / higher interest rate). Otherwise, **Approved for Immediate Disbursement**.

---

## 📁 Repository Structure

```
ps/
├── main.py                     # FastAPI core service & routing
├── schemas.py                  # Pydantic v2 data models & validation
├── train.py                    # Training & artifact serialization script
├── models/
│   └── loan_model_bundle.joblib# Pre-trained ML weights, scalers & encoders
├── test_api.py                 # Automated endpoint verification test suite
├── requirements.txt            # Python dependencies (FastAPI, XGBoost, etc.)
├── render.yaml                 # Render infrastructure-as-code blueprint
├── Dockerfile                  # Containerized deployment specification
├── .gitignore                  # Git exclusions (ignores 50MB+ raw datasets)
├── kaggle_fastapi_deployment.ipynb # Ready-to-run Kaggle notebook
├── Loan_Prediction_Student_Project.ipynb # Original student project notebook
└── README.md                   # Complete deployment & usage documentation
```

---

## 🚀 Option 1: Deploy on Render (Cloud Web Service)

Render provides free cloud hosting for FastAPI applications with automatic HTTPS.

### Step-by-Step Render Deployment:
1. **Initialize Git and Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Loan Decision FastAPI"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
   *(Note: The `.gitignore` automatically prevents pushing large 50MB+ raw CSV files to GitHub. The lightweight `loan_model_bundle.joblib` is only ~1-2 MB and is committed directly).*

2. **Create Web Service on Render:**
   - Log in to [Render.com](https://render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Select your GitHub repository.
   - Configure the following settings:
     - **Name:** `loan-decision-api`
     - **Environment:** `Python 3`
     - **Region:** Any (e.g., Oregon or Frankfurt)
     - **Branch:** `main`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Plan Type:** `Free`
   - Click **Create Web Service**.

3. **Access Your Live Cloud API:**
   - Within 2 minutes, Render assigns a live public URL (e.g., `https://loan-decision-api.onrender.com`).
   - Open `https://loan-decision-api.onrender.com/docs` to test the interactive Swagger UI!

---

## 📓 Option 2: Run Inside Kaggle

You can run this FastAPI service directly inside a Kaggle notebook and query it or expose it publicly.

1. **Upload Notebook to Kaggle:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code).
   - Click **New Notebook** $\rightarrow$ **File** $\rightarrow$ **Upload Notebook**.
   - Select `kaggle_fastapi_deployment.ipynb` from this folder.
2. **Attach Dataset (Optional):**
   - Click **Add Input** on the right sidebar and upload `loan_decision_dataset_200k.csv` (or let the notebook run with the built-in synthetic generator).
3. **Execute Cells:**
   - Run **Step 1** (Installs packages).
   - Run **Step 2 & 3** (Trains models & sets up FastAPI).
   - Run **Step 4** (Starts FastAPI in a background daemon thread at `http://127.0.0.1:8000`).
   - Run **Step 5** (Sends test requests for Applicant A & B).
   - Run **Step 6 (Optional)**: Provide a free token from [ngrok.com](https://dashboard.ngrok.com) to generate a public HTTPS link to your Kaggle instance.

---

## 💻 Option 3: Run Locally

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Models (Creates `models/loan_model_bundle.joblib`):
```bash
python train.py
```

### 3. Start the FastAPI Server:
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open Swagger UI in Browser:
- Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interactive ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 5. Run Automated Tests:
In a separate terminal:
```bash
python test_api.py
```

---

## 🔌 API Endpoints Reference

### 1. `POST /predict`
Evaluates a single applicant across both stages.

#### Request Body (`application/json`):
```json
{
  "applicant_name": "Aman Sharma",
  "applicant_annual_income": 95000.0,
  "credit_score": 760,
  "debt_to_income_ratio": 14.5,
  "loan_amount_requested": 15000.0,
  "loan_term_months": 36,
  "interest_rate": 8.5,
  "employment_experience_years": 6.0,
  "home_ownership": "MORTGAGE",
  "education_level": "Bachelor",
  "loan_purpose": "debt_consolidation",
  "revolving_utilization_rate": 25.0,
  "delinquencies_2yrs": 0,
  "inquiries_last_6mths": 0
}
```

#### Successful Response (`200 OK`):
```json
{
  "applicant_name": "Aman Sharma",
  "stage1_eligibility": {
    "decision": "Approved",
    "approval_probability_pct": 98.4,
    "meets_criteria": true,
    "reason": "Applicant satisfies underwriting criteria."
  },
  "stage2_repayment_risk": {
    "evaluated": true,
    "risk_level": "LOW RISK / SAFE BORROWER",
    "default_probability_pct": 8.2,
    "risk_drivers": []
  },
  "final_action": "APPROVED FOR IMMEDIATE DISBURSEMENT",
  "summary": "Stage 1 Approved (98.4% confidence). Stage 2 indicates strong repayment likelihood (8.2% default probability). Approved for disbursement."
}
```

### 2. `POST /batch-predict`
Evaluates a list of applicants in a single call.
```json
{
  "applicants": [
    { ...applicant_1... },
    { ...applicant_2... }
  ]
}
```

### 3. `GET /health`
Returns service and model status:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_source": "pre_trained_bundle",
  "version": "1.0.0"
}
```

---

## 🎓 Viva-Voce Key Talking Points (For Exams & Presentation)

1. **Why Two Stages instead of One?**
   Banks must first decide whether to approve an applicant. Repayment status is only observed for disbursed loans. Training on rejected applicants for repayment risk causes target leakage.
2. **Why FastAPI over Flask / Django?**
   FastAPI uses Python type hints and Pydantic for automatic data validation, generates live OpenAPI/Swagger documentation, and operates asynchronously (`async/await`) for high concurrency.
3. **Why serialize into `.joblib` instead of training in the cloud?**
   Free hosting platforms like Render have a 512MB RAM cap. Training on a 200,000-row dataset during server boot would exceed memory limits and cause timeouts. Serialized joblib weights load in under 100ms and use minimal memory.
