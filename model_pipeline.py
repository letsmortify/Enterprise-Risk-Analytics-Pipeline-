import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import shap

def run_risk_pipeline():
    print("=" * 60)
    print("STARTING ENTERPRISE RISK MODEL, PSI & SHAP EXPLAINABILITY PIPELINE")
    print("=" * 60)

    # 1. Simulate Historical Portfolio Data (1000 SMEs)
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Current_Ratio': np.random.uniform(0.8, 2.5, n_samples),
        'DSCR': np.random.uniform(0.9, 2.2, n_samples),
        'NACH_Bounces_3M': np.random.poisson(0.5, n_samples),
        'ADB_Erosion_Pct': np.random.uniform(0.0, 0.6, n_samples)
    })
    
    # Define Default (1 = Bad, 0 = Good) based on risk factors
    logit = -2.0 - 1.2*data['DSCR'] - 0.8*data['Current_Ratio'] + 1.5*data['NACH_Bounces_3M'] + 2.0*data['ADB_Erosion_Pct']
    prob = 1 / (1 + np.exp(-logit))
    data['Default_Flag'] = (prob > np.percentile(prob, 85)).astype(int)

    # 2. Train-Test Split (80/20 Stratified)
    X = data.drop(columns=['Default_Flag'])
    y = data['Default_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Fit Champion Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Model Evaluation (AUC-ROC & Gini)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    gini_score = 2 * auc_score - 1

    print(f"\n[MODEL PERFORMANCE]")
    print(f" -> Validation ROC-AUC Score: {auc_score:.4f}")
    print(f" -> Validation Gini Coefficient: {gini_score:.4f}")

    # 5. Production Monitoring: PSI (Population Stability Index) Drift
    print("\n[PRODUCTION MONITORING]")
    baseline_dscr = X_train['DSCR']
    production_dscr = np.random.uniform(0.5, 1.8, 200) # Economic Stress Drift

    bins = np.percentile(baseline_dscr, np.linspace(0, 100, 6))
    base_counts, _ = np.histogram(baseline_dscr, bins=bins)
    prod_counts, _ = np.histogram(production_dscr, bins=bins)

    base_pct = base_counts / len(baseline_dscr)
    prod_pct = np.where(prod_counts == 0, 0.0001, prod_counts / len(production_dscr))

    psi_val = np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct))
    print(f" -> Calculated Portfolio PSI (DSCR Feature): {psi_val:.4f}")

    if psi_val > 0.25:
        print(" -> ALERT: High Population Drift Detected (PSI > 0.25)! Trigger Re-calibration.")
    elif psi_val > 0.10:
        print(" -> WARNING: Moderate Population Drift (0.10 < PSI <= 0.25). Monitor closely.")
    else:
        print(" -> STATUS: Portfolio Stable (PSI <= 0.10).")

    # 6. Model Governance: SHAP Explainability Engine
    print("\n[MODEL GOVERNANCE & EXPLAINABILITY (SHAP)]")
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Calculate Mean Absolute SHAP values for global feature ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_SHAP_Impact': mean_abs_shap
    }).sort_values(by='Mean_SHAP_Impact', ascending=False)

    print(" -> Global Feature Risk Attribution (Top Drivers of Credit Score):")
    for idx, row in feature_importance.iterrows():
        print(f"    - {row['Feature']:<18}: SHAP Impact = {row['Mean_SHAP_Impact']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTED & SHAP AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_risk_pipeline()
