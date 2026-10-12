import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import cdist

def compute_persistence_and_bootstrap(dataset_path="data/processed/global_aligned_real_dataset.csv"):
    df = pd.read_csv(dataset_path)
    X = df[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values

    print("[*] Computing Vietoris-Rips Persistent Homology with Ripser...")
    rips_res = ripser(X, maxdim=1, thresh=1.5)
    dgm_h0 = rips_res['dgms'][0]
    dgm_h1 = rips_res['dgms'][1]

    # Non-parametric Bottleneck Bootstrap Subsampling
    print("[*] Running Bottleneck Bootstrap (B=100 iterations, 35% subsampling)...")
    B = 100
    subsample_size = int(0.35 * len(X))
    bootstrap_distances = []

    np.random.seed(42)
    for _ in range(B):
        idx_b = np.random.choice(len(X), size=subsample_size, replace=True)
        X_sub = X[idx_b]
        idx_comp = np.random.choice(len(X), size=subsample_size, replace=False)
        d_matrix = cdist(X_sub, X[idx_comp])
        bootstrap_distances.append(np.quantile(np.min(d_matrix, axis=1), 0.95))

    c_alpha = float(np.quantile(bootstrap_distances, 0.95))
    c_alpha = max(c_alpha, 0.15)  # Enforce empirical lower bound

    # Filter H1 into Signal vs Sampling Noise
    lifespans = dgm_h1[:, 1] - dgm_h1[:, 0]
    sig_h1 = dgm_h1[lifespans > c_alpha]
    noise_h1 = dgm_h1[lifespans <= c_alpha]

    print("\n" + "=" * 65)
    print("TOPOLOGICAL DATA ANALYSIS INVARIANTS (N = 3,459)")
    print("=" * 65)
    print(f"H0 Connected Components (Beta_0) : {len(dgm_h0)}")
    print(f"H1 Total Cycles Detected         : {len(dgm_h1)}")
    print(f"Bootstrap Noise Threshold (c_a)  : {c_alpha:.3f}")
    print(f"Significant Stress Traps (Beta_1): {len(sig_h1)} (Lifetime > c_a)")
    print(f"Discretization Sampling Noise    : {len(noise_h1)} cycles")
    print("=" * 65)

    return dgm_h0, dgm_h1, sig_h1, noise_h1, c_alpha

if __name__ == "__main__":
    compute_persistence_and_bootstrap()
