import numpy as np
import pandas as pd

def calculate_woe_iv(df, feature, target):
    # Bin attributes safely into deciles
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
    
    grouped = df.groupby('bin', observed=False).agg(
        total_accounts=(target, 'count'),
        defaults=(target, 'sum')
    )
    grouped['non_defaults'] = grouped['total_accounts'] - grouped['defaults']
    
    total_defaults = grouped['defaults'].sum()
    total_non_defaults = grouped['non_defaults'].sum()
    
    # Distribution ratios
    grouped['prob_default'] = grouped['defaults'] / total_defaults
    grouped['prob_non_default'] = grouped['non_defaults'] / total_non_defaults
    
    # Pure mathematical WoE & IV implementations (with epsilon safety adjustments)
    grouped['woe'] = np.log((grouped['prob_non_default'] + 1e-6) / (grouped['prob_default'] + 1e-6))
    grouped['iv_contribution'] = (grouped['prob_non_default'] - grouped['prob_default']) * grouped['woe']
    
    total_iv = grouped['iv_contribution'].sum()
    return grouped, total_iv

if __name__ == "__main__":
    df = pd.read_csv("corporate_loan_portfolio.csv")
    
    for metric in ['leverage_ratio', 'debt_service_risk_score', 'portfolio_vintage_months']:
        woe_df, iv_value = calculate_woe_iv(df, metric, 'default_flag')
        print(f"\n==================================================================")
        print(f"Metric: {metric} | Total Predictive Information Value (IV): {iv_value:.4f}")
        print(f"==================================================================")
        print(woe_df[['total_accounts', 'defaults', 'woe']])
