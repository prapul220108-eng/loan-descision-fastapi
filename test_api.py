"""
Test Script for Loan Decision FastAPI Endpoints
Run locally or in CI/CD:
    python test_api.py
"""

import sys
import json

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"


def test_endpoints():
    print("=" * 65)
    print("TESTING FASTAPI LOAN DECISION SYSTEM")
    print("=" * 65)

    # 1. Test Health Endpoint
    print("\n[1] Testing GET /health ...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {r.status_code}")
        print("Response:", json.dumps(r.json(), indent=2))
        assert r.status_code == 200, "Health check failed"
    except requests.exceptions.ConnectionError:
        print(f"Error: Server not running at {BASE_URL}. Start it first using:\n  uvicorn main:app --reload")
        return

    # 2. Test Model Info Endpoint
    print("\n[2] Testing GET /model-info ...")
    r = requests.get(f"{BASE_URL}/model-info", timeout=5)
    print(f"Status Code: {r.status_code}")
    print("Response:", json.dumps(r.json(), indent=2))

    # 3. Test Applicant A (Strong Profile -> Expect Approved & Low Risk)
    applicant_a = {
        "applicant_name": "Aman Sharma (Prime Applicant)",
        "applicant_annual_income": 95000,
        "credit_score": 760,
        "debt_to_income_ratio": 14.5,
        "loan_amount_requested": 15000,
        "loan_term_months": 36,
        "interest_rate": 8.5,
        "employment_experience_years": 6,
        "home_ownership": "MORTGAGE",
        "education_level": "Bachelor",
        "loan_purpose": "debt_consolidation",
        "revolving_utilization_rate": 25.0,
        "delinquencies_2yrs": 0,
        "inquiries_last_6mths": 0
    }

    print("\n[3] Testing POST /predict for Applicant A (Prime Profile) ...")
    r = requests.post(f"{BASE_URL}/predict", json=applicant_a, timeout=10)
    print(f"Status Code: {r.status_code}")
    print("Decision Response:")
    print(json.dumps(r.json(), indent=2))

    # 4. Test Applicant B (High Risk Profile -> Expect Declined or High Risk)
    applicant_b = {
        "applicant_name": "Rajesh Verma (High Risk Profile)",
        "applicant_annual_income": 30000,
        "credit_score": 540,
        "debt_to_income_ratio": 44.0,
        "loan_amount_requested": 25000,
        "loan_term_months": 60,
        "interest_rate": 22.0,
        "employment_experience_years": 1,
        "home_ownership": "RENT",
        "education_level": "High School",
        "loan_purpose": "small_business",
        "revolving_utilization_rate": 88.0,
        "delinquencies_2yrs": 2,
        "inquiries_last_6mths": 4
    }

    print("\n[4] Testing POST /predict for Applicant B (Subprime Profile) ...")
    r = requests.post(f"{BASE_URL}/predict", json=applicant_b, timeout=10)
    print(f"Status Code: {r.status_code}")
    print("Decision Response:")
    print(json.dumps(r.json(), indent=2))

    # 5. Test Batch Prediction
    print("\n[5] Testing POST /batch-predict ...")
    batch_payload = {"applicants": [applicant_a, applicant_b]}
    r = requests.post(f"{BASE_URL}/batch-predict", json=batch_payload, timeout=10)
    print(f"Status Code: {r.status_code}")
    print("Batch Response Summary:")
    res = r.json()
    print(f"Total: {res['total_processed']} | Approved: {res['approved_count']} | Rejected: {res['rejected_count']}")

    print("\n" + "=" * 65)
    print("ALL API TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_endpoints()
