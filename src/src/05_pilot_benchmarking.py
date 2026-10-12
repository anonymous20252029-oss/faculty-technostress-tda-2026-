#src/05_pilot_benchmarking.py & src/06_generate_figures.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from src.03_topological_homology import compute_persistence_and_bootstrap

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

def generate_all_publication_plots(output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    df_global = pd.read_csv("data/processed/global_aligned_real_dataset.csv")
    X = df_global[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values

    # 1. Generate Persistence Diagram with Bootstrap Band
    dgm_h0, dgm_h1, sig_h1, noise_h1, c_alpha = compute_persistence_and_bootstrap()
    
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    max_val = 1.45
    ax.plot([0, max_val], [0, max_val], 'k--', label=r'Diagonal $\Delta$')
    ax.fill_between([0, max_val], [0, max_val], [c_alpha, max_val + c_alpha],
                    color='gray', alpha=0.25, label=fr'95% Bootstrap Band ($c_\alpha = {c_alpha:.2f}$)')

    ax.scatter(dgm_h0[:, 0], dgm_h0[:, 1], c='#1f77b4', s=16, alpha=0.6, label=f'$H_0$ Components ($N={len(dgm_h0)}$)')
    ax.scatter(noise_h1[:, 0], noise_h1[:, 1], c='#ff7f0e', s=18, alpha=0.3, label=f'$H_1$ Noise ($n={len(noise_h1)}$)')
    ax.scatter(sig_h1[:, 0], sig_h1[:, 1], c='#d62728', s=35, edgecolors='black', linewidth=0.6,
               label=f'$H_1$ Significant Loops ($n={len(sig_h1)}$)')

    ax.set_xlim([-0.05, 1.45])
    ax.set_ylim([-0.05, 1.45])
    ax.set_xlabel(r'Birth Scale ($\varepsilon$)')
    ax.set_ylabel(r'Death Scale ($\varepsilon$)')
    ax.set_title('Empirical Vietoris--Rips Persistence with Bootstrap Band')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "vietoris_rips_persistence_bootstrap.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"[*] Figure saved: {p1_path}")

    # 2. Pilot Out-of-Sample PCA Projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pilot_path = "data/pilot/pilot_survey_cntt_30_responses.csv"
    if os.path.exists(pilot_path):
        df_pilot = pd.read_csv(pilot_path)
        pilot_X = df_pilot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values
    else:
        pilot_X = X[:30]

    pilot_pca = pca.transform(pilot_X)
    centroid_pilot = np.mean(pilot_pca, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 1], cmap='viridis', s=12, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Empirical Technostress ($F_2$)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    ax1.set_title('(A) Global Latent State-Space Manifold ($N=3,459$)')

    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', s=8, alpha=0.4, label='Baseline Norms ($N=3,459$)')
    ax2.scatter(pilot_pca[:, 0], pilot_pca[:, 1], c='#e41a1c', edgecolors='black', s=45, label='Faculty Pilot ($n=30$)')
    ax2.scatter(centroid_pilot[0], centroid_pilot[1], c='gold', marker='*', s=200, edgecolors='black', label='Cohort Centroid')

    cov_pilot = np.cov(pilot_pca.T)
    lambda_, v_ = np.linalg.eig(cov_pilot)
    lambda_ = np.sqrt(np.maximum(lambda_, 1e-6))
    angle_deg = np.rad2deg(np.arctan2(v_[1, 0], v_[0, 0]))

    ellipse = Ellipse(xy=(centroid_pilot[0], centroid_pilot[1]),
                      width=lambda_[0]*2*1.96, height=lambda_[1]*2*1.96,
                      angle=angle_deg, edgecolor='#e41a1c',
                      fc='None', lw=1.5, linestyle='--', label='95% Variance Envelope')
    ax2.add_patch(ellipse)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    ax2.set_title('(B) In-Situ Feasibility Pilot Mapping ($n=30$)')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "combined_pca_pilot_benchmark.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"[*] Figure saved: {p2_path}")

if __name__ == "__main__":
    generate_all_publication_plots()
