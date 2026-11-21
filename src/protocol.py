"""Общие функции протокола регулятора М5 (шаги 5–7)."""
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph


def leads_to(v: int, edge_ids, ok_mask: np.ndarray, edge_index: np.ndarray, und: torch.Tensor,
             num_nodes: int, hops: int = 2) -> bool:
    """Проверяемость: каждый поток из edge_ids за <= hops хопов от своего дальнего конца
    ведёт к транзакции с ok_mask = True. Обход не проходит через сам алерт v,
    иначе любой подходящий сосед v засчитывался бы любому потоку."""
    keep = (und[0] != v) & (und[1] != v)
    und_v = und[:, keep]
    for e in edge_ids:
        start = [int(u) for u in edge_index[:, e] if u != v]
        if not start:
            return False
        nodes, _, _, _ = k_hop_subgraph(start, hops, und_v, num_nodes=num_nodes)
        nodes = nodes.numpy()
        nodes = nodes[nodes != v]
        if not ok_mask[nodes].any():
            return False
    return True


def wilson(k: int, n: int, z: float = 1.96):
    """95% интервал Уилсона для доли k/n (занятие 11: Wilson вместо Wald при малых n и долях у края)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)
