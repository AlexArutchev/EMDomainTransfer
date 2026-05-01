import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

centroids = np.load(RESULTS_DIR / "domain_centroids.npy")
with open(RESULTS_DIR / "domain_names.json") as f:
    domain_names = json.load(f)

sim = centroids @ centroids.T
dist = 1 - sim
np.fill_diagonal(dist, 0.0)

df = pd.DataFrame(dist, index=domain_names, columns=domain_names)

print("--- Pairwise Cosine Distance Matrix ---")
print(df.to_string(float_format="{:.4f}".format))

g = sns.clustermap(
    df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    figsize=(12, 10),
    linewidths=0.5,
    method="average",
    metric="euclidean",
)
g.fig.suptitle("Pairwise Cosine Distance Between Domain Centroids",
               fontsize=14, y=1.02)
plt.savefig(FIGURES_DIR / "domain_embedding_distances.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nHeatmap saved → {FIGURES_DIR / 'domain_embedding_distances.png'}")

pairs = [
    (dist[i, j], domain_names[i], domain_names[j])
    for i in range(len(domain_names))
    for j in range(i + 1, len(domain_names))
]
pairs.sort()
avg_dist = np.mean([d for d, _, _ in pairs])

print("\n3 most similar domain pairs (lowest distance):")
for d, a, b in pairs[:3]:
    print(f"  {a} <-> {b}: {d:.4f}")

print("\n3 most dissimilar domain pairs (highest distance):")
for d, a, b in pairs[-3:][::-1]:
    print(f"  {a} <-> {b}: {d:.4f}")

print(f"\nAverage pairwise distance: {avg_dist:.4f}")
