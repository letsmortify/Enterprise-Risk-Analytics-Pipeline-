import numpy as np
import pandas as pd

def generate_risk_portfolio(n_samples=50000, random_state=42):
    np.random.seed(random_state)
    print(f"Initializing synthetic generation for {n_samples} corporate entities...")
    
    corporate_ids = [f"CORP-{i:05d}" for i in range(1, n_samples + 1)]
    sectors = np.random.choice(['Manufacturing', 'Technology', 'Retail', 'Infrastructure', 'Healthcare'], 
                               size=n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    
    # Financial drivers (Log-normal to simulate realistic corporate distributions)
    leverage_ratio = np.random.lognormal(mean=0.8, sigma=0.4, size=n_samples)      # Total Debt / Equity
    interest_coverage = np.random.lognormal(mean=1.2, sigma=0.5, size=n_samples)   # EBITDA / Interest Expense
    interest_coverage = np.clip(15 / (interest_coverage + 0.1), 0.5, 40)           # Higher = higher risk
    
    vintage_months = np.random.randint(1, 60, size=n_samples)
    
    # Latent risk formula blending features
    latent_risk = (0.4 * leverage_ratio) + (0.5 * interest_coverage) - (0.01 * vintage_months)
    latent_risk += np.random.normal(0, 0.5, size=n_samples)
    
    # Hard hurdle to isolate high-risk defaults (~2.5% default rate)
    threshold = np.percentile(latent_risk, 97.5)
    default_flag = (latent_risk >= threshold).astype(int)
    
    df = pd.DataFrame({
        'corporate_id': corporate_ids,
        'industry_sector': sectors,
        'leverage_ratio': leverage_ratio,
        'debt_service_risk_score': interest_coverage,
        'portfolio_vintage_months': vintage_months,
        'default_flag': default_flag
    })
    
    print(f"Generation complete. Portfolio Default Rate: {df['default_flag'].mean()*100:.2f}%")
    return df

if __name__ == "__main__":
    portfolio = generate_risk_portfolio()
    portfolio.to_csv("corporate_loan_portfolio.csv", index=False)

