"""Синтетика в формате сырых файлов Elliptic — для проверки кода без скачивания данных.
python tests/make_fake_elliptic.py data/fake_elliptic --per-step 400"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--per-step", type=int, default=400)
a = ap.parse_args()
rng = np.random.default_rng(0)
raw = a.root / "raw"
raw.mkdir(parents=True, exist_ok=True)
n_steps, k = 49, a.per_step
N = n_steps * k
tx = rng.permutation(np.arange(10**6, 10**6 + N))
ts = np.repeat(np.arange(1, n_steps + 1), k)
illicit = rng.random(N) < 0.1
X = rng.normal(size=(N, 165))
X[:, :5] += illicit[:, None] * 1.0
edges = []
for s in range(n_steps):
    idx = np.arange(s * k, (s + 1) * k)
    src = rng.choice(idx, 2 * k)
    dst = np.where(illicit[src] & (rng.random(2 * k) < 0.7),
                   rng.choice(idx[illicit[idx]], 2 * k), rng.choice(idx, 2 * k))
    keep = src != dst
    edges.append(np.stack([src[keep], dst[keep]], 1))
E = np.unique(np.concatenate(edges), axis=0)
cls = np.where(illicit, "1", "2").astype(object)
cls[rng.random(N) < 0.77] = "unknown"
pd.DataFrame(np.column_stack([tx, ts, X])).astype({0: int, 1: int}).to_csv(raw / "elliptic_txs_features.csv", header=False, index=False)
pd.DataFrame({"txId1": tx[E[:, 0]], "txId2": tx[E[:, 1]]}).to_csv(raw / "elliptic_txs_edgelist.csv", index=False)
pd.DataFrame({"txId": tx, "class": cls}).to_csv(raw / "elliptic_txs_classes.csv", index=False)
print(f"{N} узлов, {len(E)} рёбер -> {raw}")
