import kmapper as km
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def build_mapper_complex(dataset_path="data/processed/global_aligned_real_dataset.csv", output_html="mapper_topology.html"):
    df = pd.read_csv(dataset_path)
    X = df[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values

    mapper = km.KeplerMapper(verbose=1)
    projected_data = mapper.fit_transform(X, projection=PCA(n_components=2))

    # Construct nerve simplicial complex
    graph = mapper.map(
        projected_data,
        X,
        cover=km.Cover(n_cubes=12, perc_overlap=0.40),
        clusterer=DBSCAN(eps=0.50, min_samples=3)
    )

    print(f"[*] Kepler-Mapper Complex: |V| = {len(graph['nodes'])} hyper-nodes, |E| = {sum(len(v) for v in graph['links'].values()) // 2} edges")
    mapper.visualize(graph, path_html=output_html, title="Cognitive Technostress Manifold")
    print(f"[*] Interactive HTML graph saved to {output_html}")
    return graph

if __name__ == "__main__":
    build_mapper_complex()
