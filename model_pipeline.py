import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import shap

def run_risk_pipeline():
    print("=" * 65)
    print("STARTING ENTERPRISE RISK, SHAP & STRESS TESTING PIPELINE")
    print("=" * 65)

    # 1. Simulate Historical Portfolio Data (1000 SMEs)
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Current_Ratio': np.random.uniform(0.8, 2.5, n_samples),
        'DSCR': np.random.uniform(0.9, 2.2, n_samples),
        'NACH_Bounces_3M': np.random.poisson(0.5, n_samples),
        'ADB_Erosion_Pct': np.random.uniform(0.0, 0.6, n_samples),
        'EAD': np.random.uniform(100000, 5000000, n_samples), # Exposure at Default ($)
        'LGD': np.random.uniform(0.30, 0.50, n_samples)      # Loss Given Default (%)
    })
    
    # Define Default (1 = Bad, 0 = Good) based on risk factors
    logit = -2.0 - 1.2*data['DSCR'] - 0.8*data['Current_Ratio'] + 1.5*data['NACH_Bounces_3M'] + 2.0*data['ADB_Erosion_Pct']
    prob = 1 / (1 + np.exp(-logit))
    data['Default_Flag'] = (prob > np.percentile(prob, 85)).astype(int)

    # 2. Train-Test Split (80/20 Stratified)
    features = ['Current_Ratio', 'DSCR', 'NACH_Bounces_3M', 'ADB_Erosion_Pct']
    X = data[features]
    y = data['Default_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Fit Champion Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Model Evaluation (AUC-ROC & Gini)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    gini_score = 2 * auc_score - 1

    print(f"\n[1. MODEL PERFORMANCE]")
    print(f" -> Validation ROC-AUC Score: {auc_score:.4f}")
    print(f" -> Validation Gini Coefficient: {gini_score:.4f}")

    # 5. Production Monitoring: PSI (Population Stability Index) Drift
    print("\n[2. PRODUCTION MONITORING]")
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
        print(" -> ALERT: High Population Drift Detected (PSI > 0.25)!")
    else:
        print(" -> STATUS: Portfolio Drift Within Normal Bounds.")

    # 6. Model Governance: SHAP Explainability Engine
    print("\n[3. MODEL GOVERNANCE (SHAP)]")
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_SHAP_Impact': mean_abs_shap
    }).sort_values(by='Mean_SHAP_Impact', ascending=False)

    for idx, row in feature_importance.iterrows():
        print(f"    - {row['Feature']:<18}: SHAP Impact = {row['Mean_SHAP_Impact']:.4f}")

    # 7. MACROECONOMIC STRESS TESTING & IFRS 9 ECL PROVISIONING
    print("\n[4. IFRS 9 MACRO STRESS TESTING & ECL PROVISIONING]")
    test_df = data.loc[X_test.index].copy()
    test_df['Baseline_PD'] = y_pred_proba

    # Apply Macro Shock: Scenario = GDP drops by 2%, Interest Rate hikes by 150bps
    # Shift log-odds proportional to macro severity multiplier (1.45x risk expansion)
    stressed_logit = logit[X_test.index] + np.log(1.45)
    test_df['Stressed_PD'] = 1 / (1 + np.exp(-stressed_logit))

    # SICR Trigger: If Stressed PD increases by > 50% relative to baseline, shift to Stage 2 (Lifetime ECL)
    test_df['SICR_Flag'] = (test_df['Stressed_PD'] / test_df['Baseline_PD'] > 1.50).astype(int)

    # ECL Calculations ($)
    # Stage 1: 12-Month ECL | Stage 2: Lifetime ECL (Simulated as 2.8x 12M ECL duration multiplier)
    test_df['Baseline_ECL'] = test_df['Baseline_PD'] * test_df['LGD'] * test_df['EAD']
    test_df['Stressed_ECL'] = np.where(
        test_df['SICR_Flag'] == 1,
        test_df['Stressed_PD'] * test_df['LGD'] * test_df['EAD'] * 2.8, # Lifetime multiplier
        test_df['Stressed_PD'] * test_df['LGD'] * test_df['EAD']
    )

    total_base_ecl = test_df['Baseline_ECL'].sum()
    total_stress_ecl = test_df['Stressed_ECL'].sum()
    stage_2_count = test_df['SICR_Flag'].sum()

    print(f" -> Baseline Portfolio ECL Provision : ${total_base_ecl:,.2f}")
    print(f" -> Stressed Portfolio ECL Provision : ${total_stress_ecl:,.2f}")
    print(f" -> Provision Capital Impact           : +{((total_stress_ecl - total_base_ecl) / total_base_ecl)*100:.2f}%")
    print(f" -> Accounts Shifted to Stage 2 (SICR) : {stage_2_count} / {len(test_df)} ({stage_2_count/len(test_df)*100:.1f}%)")

    print("\n" + "=" * 65)
    print("COMPLETE END-TO-END PIPELINE EXECUTED & SYNCED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    run_risk_pipeline()
