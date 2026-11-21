from typing import List, Optional
from pydantic import BaseModel, Field


class ApplicantRequest(BaseModel):
    applicant_name: Optional[str] = Field(
        default="Applicant",
        description="Full name of the loan applicant",
        example="Aman Sharma"
    )
    applicant_annual_income: float = Field(
        ...,
        gt=0,
        description="Annual gross income in USD",
        example=95000.0
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="FICO credit score (300 - 850)",
        example=760
    )
    debt_to_income_ratio: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Debt-to-Income (DTI) ratio percentage",
        example=14.5
    )
    loan_amount_requested: float = Field(
        ...,
        gt=0,
        description="Requested principal loan amount in USD",
        example=15000.0
    )
    loan_term_months: int = Field(
        default=36,
        description="Loan repayment duration in months (e.g., 36, 60)",
        example=36
    )
    interest_rate: float = Field(
        default=10.5,
        ge=0.0,
        le=40.0,
        description="Estimated annual loan interest rate (%)",
        example=8.5
    )
    employment_experience_years: float = Field(
        default=3.0,
        ge=0.0,
        description="Total employment experience in years",
        example=6.0
    )
    home_ownership: str = Field(
        default="MORTGAGE",
        description="Home ownership status: MORTGAGE, RENT, OWN, OTHER",
        example="MORTGAGE"
    )
    education_level: str = Field(
        default="Bachelor",
        description="Highest education attained: Bachelor, Master, High School, Associate, Doctorate",
        example="Bachelor"
    )
    loan_purpose: str = Field(
        default="debt_consolidation",
        description="Purpose of loan: debt_consolidation, credit_card, home_improvement, small_business, car, medical, major_purchase, other",
        example="debt_consolidation"
    )
    revolving_utilization_rate: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Credit card revolving utilization rate (%)",
        example=25.0
    )
    delinquencies_2yrs: int = Field(
        default=0,
        ge=0,
        description="Number of past-due payments (30+ days) in past 2 years",
        example=0
    )
    inquiries_last_6mths: int = Field(
        default=0,
        ge=0,
        description="Number of hard credit inquiries in past 6 months",
        example=0
    )

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class Stage1Result(BaseModel):
    decision: str = Field(..., description="'Approved' or 'Rejected'")
    approval_probability_pct: float = Field(..., description="Approval probability score (0-100%)")
    meets_criteria: bool = Field(..., description="Whether borrower passed Stage 1 eligibility threshold")
    reason: str = Field(..., description="Explanation for Stage 1 decision")


class Stage2Result(BaseModel):
    evaluated: bool = Field(..., description="True if Stage 2 was performed (only for approved loans)")
    risk_level: str = Field(..., description="'LOW RISK / SAFE BORROWER', 'HIGH RISK OF DEFAULT', or 'N/A'")
    default_probability_pct: Optional[float] = Field(None, description="Estimated probability of default (0-100%)")
    risk_drivers: List[str] = Field(default_factory=list, description="Key metrics influencing the risk profile")


class LoanPredictionResponse(BaseModel):
    applicant_name: str
    stage1_eligibility: Stage1Result
    stage2_repayment_risk: Stage2Result
    final_action: str = Field(
        ...,
        description="'APPROVED FOR IMMEDIATE DISBURSEMENT', 'CONDITIONAL APPROVAL (Require guarantor or higher interest rate)', or 'DECLINE APPLICATION'"
    )
    summary: str


class BatchPredictionRequest(BaseModel):
    applicants: List[ApplicantRequest]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    approved_count: int
    rejected_count: int
    results: List[LoanPredictionResponse]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    version: str
