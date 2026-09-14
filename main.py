from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import shap

app = FastAPI(
    title="Enterprise Credit Risk & Regulatory Intelligence API",
    description="Production REST API for Credit Decisioning, SHAP Explainability, and IFRS 9 Stress Testing",
    version="1.0.0"
)

# --- 1. TRAIN BASELINE IN-MEMORY MODEL (Simulated Model Artifact) ---
np.random.seed(42)
n_samples = 1000
train_data = pd.DataFrame({
    'Current_Ratio': np.random.uniform(0.8, 2.5, n_samples),
    'DSCR': np.random.uniform(0.9, 2.2, n_samples),
    'NACH_Bounces_3M': np.random.poisson(0.5, n_samples),
    'ADB_Erosion_Pct': np.random.uniform(0.0, 0.6, n_samples)
})
logit = -2.0 - 1.2*train_data['DSCR'] - 0.8*train_data['Current_Ratio'] + 1.5*train_data['NACH_Bounces_3M'] + 2.0*train_data['ADB_Erosion_Pct']
prob = 1 / (1 + np.exp(-logit))
y_train = (prob > np.percentile(prob, 85)).astype(int)

model = LogisticRegression()
model.fit(train_data, y_train)
explainer = shap.Explainer(model, train_data)


# --- 2. PYDANTIC INPUT/OUTPUT SCHEMAS ---
class LoanApplication(BaseModel):
    borrower_id: str = Field(..., example="CORP_SME_8841")
    current_ratio: float = Field(..., gt=0, example=1.45)
    dscr: float = Field(..., gt=0, example=1.15)
    nach_bounces_3m: int = Field(..., ge=0, example=2)
    adb_erosion_pct: float = Field(..., ge=0, le=1.0, example=0.25)
    ead: float = Field(..., gt=0, example=1500000.0) # Exposure at Default ($)
    lgd: float = Field(0.40, ge=0, le=1.0, example=0.40) # Loss Given Default (%)

class CreditDecisionResponse(BaseModel):
    borrower_id: str
    baseline_pd: float
    credit_decision: str
    shap_attribution: dict
    stressed_pd: float
    sicr_flag: int
    baseline_ecl: float
    stressed_ecl: float


# --- 3. REST API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "ONLINE", "system": "Credit Risk & Regulatory Engine"}

@app.post("/predict", response_model=CreditDecisionResponse)
def evaluate_loan(applicant: LoanApplication):
    try:
        # Prepare Feature Vector
        features = pd.DataFrame([{
            'Current_Ratio': applicant.current_ratio,
            'DSCR': applicant.dscr,
            'NACH_Bounces_3M': applicant.nach_bounces_3m,
            'ADB_Erosion_Pct': applicant.adb_erosion_pct
        }])

        # Baseline Prediction
        pd_val = float(model.predict_proba(features)[0, 1])
        decision = "REJECT" if pd_val > 0.15 else "APPROVE"

        # SHAP Attribution
        shap_vals = explainer(features)
        shap_dict = {col: float(val) for col, val in zip(features.columns, shap_vals.values[0])}

        # Macroeconomic Stress Test & IFRS 9 ECL Calculation
        base_log_odds = np.log(pd_val / (1 - pd_val + 1e-15))
        stressed_log_odds = base_log_odds + np.log(1.45) # 1.45x macro shock multiplier
        stressed_pd_val = float(1 / (1 + np.exp(-stressed_log_odds)))

        sicr = 1 if (stressed_pd_val / pd_val) > 1.20 else 0
        
        base_ecl = pd_val * applicant.lgd * applicant.ead
        stressed_ecl = (stressed_pd_val * applicant.lgd * applicant.ead * 2.8) if sicr == 1 else (stressed_pd_val * applicant.lgd * applicant.ead)

        return CreditDecisionResponse(
            borrower_id=applicant.borrower_id,
            baseline_pd=round(pd_val, 4),
            credit_decision=decision,
            shap_attribution=shap_dict,
            stressed_pd=round(stressed_pd_val, 4),
            sicr_flag=sicr,
            baseline_ecl=round(base_ecl, 2),
            stressed_ecl=round(stressed_ecl, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
