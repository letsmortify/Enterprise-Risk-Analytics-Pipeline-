import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import shap

def run_risk_pipeline():
    print("=" * 65)
    print("STARTING ENTERPRISE RISK, SHAP VISUALIZATION & STRESS PIPELINE")
    print("=" * 65)

    # 1. Simulate Historical Portfolio Data (1000 SMEs)
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Current_Ratio': np.random.uniform(0.8, 2.5, n_samples),
        'DSCR': np.random.uniform(0.9, 2.2, n_samples),
        'NACH_Bounces_3M': np.random.poisson(0.5, n_samples),
        'ADB_Erosion_Pct': np.random.uniform(0.0, 0.6, n_samples),
        'EAD': np.random.uniform(100000, 5000000, n_samples),
        'LGD': np.random.uniform(0.30, 0.50, n_samples)
    })
    
    # Target definition
    logit = -2.0 - 1.2*data['DSCR'] - 0.8*data['Current_Ratio'] + 1.5*data['NACH_Bounces_3M'] + 2.0*data['ADB_Erosion_Pct']
    prob = 1 / (1 + np.exp(-logit))
    data['Default_Flag'] = (prob > np.percentile(prob, 85)).astype(int)

    # 2. Train-Test Split
    features = ['Current_Ratio', 'DSCR', 'NACH_Bounces_3M', 'ADB_Erosion_Pct']
    X = data[features]
    y = data['Default_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Fit Champion Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Model Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    gini_score = 2 * auc_score - 1

    print(f"\n[1. MODEL PERFORMANCE]")
    print(f" -> Validation ROC-AUC Score: {auc_score:.4f}")
    print(f" -> Validation Gini Coefficient: {gini_score:.4f}")

    # 5. Production Monitoring (PSI)
    print("\n[2. PRODUCTION MONITORING]")
    baseline_dscr = X_train['DSCR']
    production_dscr = np.random.uniform(0.5, 1.8, 200)

    bins = np.percentile(baseline_dscr, np.linspace(0, 100, 6))
    base_counts, _ = np.histogram(baseline_dscr, bins=bins)
    prod_counts, _ = np.histogram(production_dscr, bins=bins)

    base_pct = base_counts / len(baseline_dscr)
    prod_pct = np.where(prod_counts == 0, 0.0001, prod_counts / len(production_dscr))

    psi_val = np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct))
    print(f" -> Calculated Portfolio PSI (DSCR Feature): {psi_val:.4f}")

    # 6. SHAP Governance & Automated Plot Export
    print("\n[3. MODEL GOVERNANCE & AUTOMATED SHAP VISUALIZATION]")
    explainer = shap.Explainer(model, X_train)
    shap_explanation = explainer(X_test)

    # Extract high-risk borrower (highest predicted PD in test set)
    test_df = data.loc[X_test.index].copy()
    test_df['Baseline_PD'] = y_pred_proba
    high_risk_idx = test_df['Baseline_PD'].idxmax()
    test_pos = X_test.index.get_loc(high_risk_idx)

    # Generate and save SHAP Waterfall plot
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_explanation[test_pos], show=False)
    plot_filename = "shap_waterfall_borrower.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Isolated High-Risk Borrower ID : Index #{high_risk_idx} (PD = {test_df.loc[high_risk_idx, 'Baseline_PD']:.2%})")
    print(f" -> SHAP Waterfall Plot Exported   : {plot_filename}")

    # 7. IFRS 9 Macro Stress Testing
    print("\n[4. IFRS 9 MACRO STRESS TESTING & ECL PROVISIONING]")
    base_log_odds = np.log(test_df['Baseline_PD'] / (1 - test_df['Baseline_PD'] + 1e-15))
    stressed_log_odds = base_log_odds + np.log(1.45)
    test_df['Stressed_PD'] = 1 / (1 + np.exp(-stressed_log_odds))

    test_df['SICR_Flag'] = (test_df['Stressed_PD'] / test_df['Baseline_PD'] > 1.20).astype(int)

    test_df['Baseline_ECL'] = test_df['Baseline_PD'] * test_df['LGD'] * test_df['EAD']
    test_df['Stressed_ECL'] = np.where(
        test_df['SICR_Flag'] == 1,
        test_df['Stressed_PD'] * test_df['LGD'] * test_df['EAD'] * 2.8,
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
    print("PIPELINE EXECUTED & SHAP WATERFALL PLOT SYNCED")
    print("=" * 65)

if __name__ == "__main__":
    run_risk_pipeline()
