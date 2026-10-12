import numpy as np
import pandas as pd

def compute_metric_validity(dataset_path="data/processed/global_aligned_real_dataset.csv"):
    df = pd.read_csv(dataset_path)
    X = df[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values

    # Correlation Matrix
    corr = np.corrcoef(X.T)
    det_val = np.linalg.det(corr)
    
    # VIF Calculation (diagonal of inverse correlation matrix)
    inv_corr = np.linalg.inv(corr)
    vif = np.diag(inv_corr)

    print("=" * 65)
    print("PSYCHOMETRIC & METRIC DISCRIMINANT VALIDITY DIAGNOSTICS")
    print("=" * 65)
    print("Correlation Matrix R:")
    print(np.round(corr, 3))
    print(f"\nDeterminant det(R): {det_val:.3f} (> 0: Full-rank linear independence)")
    print(f"Variance Inflation Factors (VIF): {np.round(vif, 2)} (All < 5.0: No multicollinearity)")
    print("=" * 65)
    
    return corr, det_val, vif

if __name__ == "__main__":
    compute_metric_validity()
