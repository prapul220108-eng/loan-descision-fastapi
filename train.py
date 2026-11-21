"""
Train and Export Stage 1 & Stage 2 Machine Learning Models
Based on B.Tech CSE Loan Eligibility & Repayment Risk Project
Outputs serialized artifact: models/loan_model_bundle.joblib
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    USE_XGBOOST = False

FEATURES = [
    'applicant_annual_income', 'credit_score', 'debt_to_income_ratio',
    'loan_amount_requested', 'loan_term_months', 'interest_rate',
    'employment_experience_years', 'home_ownership', 'education_level',
    'loan_purpose', 'revolving_utilization_rate', 'delinquencies_2yrs',
    'inquiries_last_6mths'
]
CAT_COLS = ['home_ownership', 'education_level', 'loan_purpose']


def train_models(data_path: str, output_path: str, sample_size: int = None):
    print(f"--> Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"--> Total records loaded: {len(df):,}")

    if sample_size and sample_size < len(df):
        print(f"--> Subsampling {sample_size:,} rows for rapid training...")
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Clean & Select Features
    df_clean = df[FEATURES + ['loan_approval_status', 'loan_repayment_status']].copy()

    # Encode Categoricals
    print("--> Fitting Categorical Label Encoders...")
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le

    # ---------------------------------------------------------
    # STAGE 1: Loan Approval Status (1 = Approved, 0 = Rejected)
    # ---------------------------------------------------------
    print("\n[STAGE 1] Training Loan Approval Model...")
    X1 = df_clean[FEATURES]
    y1 = (df_clean['loan_approval_status'] == 'Approved').astype(int)

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.20, random_state=42, stratify=y1
    )

    scaler1 = StandardScaler()
    X1_train_scaled = scaler1.fit_transform(X1_train)
    X1_test_scaled = scaler1.transform(X1_test)

    if USE_XGBOOST:
        print("  Using XGBClassifier...")
        model_stage1 = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        print("  Using HistGradientBoostingClassifier (XGBoost fallback)...")
        model_stage1 = HistGradientBoostingClassifier(random_state=42, max_iter=100)

    model_stage1.fit(X1_train_scaled, y1_train)

    y1_pred = model_stage1.predict(X1_test_scaled)
    y1_prob = model_stage1.predict_proba(X1_test_scaled)[:, 1]
    acc1 = accuracy_score(y1_test, y1_pred)
    roc1 = roc_auc_score(y1_test, y1_prob)
    f1_1 = f1_score(y1_test, y1_pred)
    print(f"  Stage 1 Evaluation -> Accuracy: {acc1*100:.2f}% | ROC-AUC: {roc1*100:.2f}% | F1: {f1_1*100:.2f}%")

    # ---------------------------------------------------------
    # STAGE 2: Repayment Default Risk (Approved loans only)
    # ---------------------------------------------------------
    print("\n[STAGE 2] Training Loan Repayment Default Risk Model...")
    df_approved = df_clean[df_clean['loan_approval_status'] == 'Approved'].dropna(subset=['loan_repayment_status'])
    X2 = df_approved[FEATURES]
    y2 = (df_approved['loan_repayment_status'] == 'Defaulted').astype(int)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.20, random_state=42, stratify=y2
    )

    scaler2 = StandardScaler()
    X2_train_scaled = scaler2.fit_transform(X2_train)
    X2_test_scaled = scaler2.transform(X2_test)

    if USE_XGBOOST:
        print("  Using XGBClassifier...")
        model_stage2 = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        print("  Using HistGradientBoostingClassifier (XGBoost fallback)...")
        model_stage2 = HistGradientBoostingClassifier(random_state=42, max_iter=100)

    model_stage2.fit(X2_train_scaled, y2_train)

    y2_pred = model_stage2.predict(X2_test_scaled)
    y2_prob = model_stage2.predict_proba(X2_test_scaled)[:, 1]
    acc2 = accuracy_score(y2_test, y2_pred)
    roc2 = roc_auc_score(y2_test, y2_prob)
    f1_2 = f1_score(y2_test, y2_pred)
    print(f"  Stage 2 Evaluation -> Accuracy: {acc2*100:.2f}% | ROC-AUC: {roc2*100:.2f}% | F1: {f1_2*100:.2f}%")

    # Package Artifacts
    artifacts = {
        'stage1_model': model_stage1,
        'scaler1': scaler1,
        'stage2_model': model_stage2,
        'scaler2': scaler2,
        'label_encoders': encoders,
        'features': FEATURES,
        'cat_cols': CAT_COLS,
        'metrics': {
            'stage1': {'accuracy': float(acc1), 'roc_auc': float(roc1), 'f1': float(f1_1)},
            'stage2': {'accuracy': float(acc2), 'roc_auc': float(roc2), 'f1': float(f1_2)}
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(artifacts, output_path, compress=3)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n--> Successfully saved model bundle to: {output_path} ({file_size_mb:.2f} MB)")
    print("--> Ready for instant deployment on FastAPI, Render, or Kaggle!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 & 2 Loan Models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_decision_dataset_200k.csv"),
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "loan_model_bundle.joblib"),
        help="Target output joblib file path"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: number of rows to sample for fast training"
    )

    args = parser.parse_args()
    train_models(args.data_path, args.output_path, args.sample_size)
