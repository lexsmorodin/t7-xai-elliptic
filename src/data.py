"""Загрузка Elliptic через PyG + шаг времени + временное разбиение."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

from . import config as C


def _meta(root: Path):
    """PyG не хранит txId и шаг времени в Data: читаем столбцы 0–1 сырого CSV и кешируем.

    PyG строит x из того же CSV без переупорядочивания строк, поэтому порядок совпадает.
    """
    cache = root / "processed" / "meta.npz"
    if not cache.exists():
        raw = root / "raw" / "elliptic_txs_features.csv"
        df = pd.read_csv(raw, header=None, usecols=[0, 1])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, tx_id=df[0].to_numpy(np.int64), time_step=df[1].to_numpy(np.int64))
    m = np.load(cache)
    return torch.from_numpy(m["tx_id"]), torch.from_numpy(m["time_step"])


def steps_mask(ts: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return (ts >= lo) & (ts <= hi)


def load_elliptic(root: Path = C.DATA_ROOT):
    """Data с полями x [N,165], y {0,1,2}, edge_index [2,E] (направленные потоки),
    tx_id [N], time_step [N], train/val/test_mask (только размеченные узлы)."""
    root = Path(root)
    data = EllipticBitcoinDataset(str(root))[0]
    data.tx_id, data.time_step = _meta(root)
    assert data.time_step.numel() == data.num_nodes, "шаг времени не совпал с числом узлов"

    labeled = data.y != C.UNKNOWN
    ts = data.time_step
    data.train_mask = labeled & steps_mask(ts, *C.TRAIN_STEPS)
    data.val_mask = labeled & steps_mask(ts, *C.VAL_STEPS)
    data.test_mask = labeled & steps_mask(ts, *C.TEST_STEPS)

    # Проверка свойства из раздела 2: рёбер между шагами времени нет
    src, dst = data.edge_index
    data.cross_step_edges = int((ts[src] != ts[dst]).sum())
    return data


def describe(data) -> str:
    y, ts = data.y, data.time_step
    lines = [
        f"узлов {data.num_nodes:,}, рёбер {data.num_edges:,}, признаков {data.num_features}, "
        f"шагов {int(ts.min())}–{int(ts.max())}, рёбер между шагами: {data.cross_step_edges}",
        f"метки: illicit {(y == C.ILLICIT).float().mean():.1%}, licit {(y == C.LICIT).float().mean():.1%}, "
        f"unknown {(y == C.UNKNOWN).float().mean():.1%}",
    ]
    for name in ("train", "val", "test"):
        m = getattr(data, f"{name}_mask")
        lines.append(f"{name:5s}: размечено {int(m.sum()):6,}, из них illicit {int((y[m] == C.ILLICIT).sum()):5,}")
    return "\n".join(lines)
