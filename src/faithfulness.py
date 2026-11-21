"""Метрики верности объяснения М1: M-1…M-6."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .explain import Neighborhood, _z_batch


@torch.no_grad()
def z_keep(model, data, nb: Neighborhood, keep: np.ndarray) -> float:
    """z(v), когда из E_v оставлены только рёбра keep (bool по переменным рёбрам)."""
    w = torch.as_tensor(keep, dtype=torch.float).unsqueeze(0)
    return float(_z_batch(model, data.x[nb.nodes], nb, w))


def fidelity(model, data, nb, order: np.ndarray, k: int, logit_tau: float):
    """M-1 / M-2 для top-k рёбер в порядке order (индексы переменных рёбер)."""
    n = int(nb.var.sum())
    full = np.ones(n, bool)
    top = np.zeros(n, bool)
    top[order[:k]] = True
    z = z_keep(model, data, nb, full)
    z_only = z_keep(model, data, nb, top)
    z_wo = z_keep(model, data, nb, ~top)
    return dict(z=z, fid_minus_kept=z_only >= logit_tau, fid_minus_ratio=z_only / z if z else np.nan,
                fid_plus=z - z_wo, fid_plus_flip=z_wo < logit_tau)


def deletion_curve(model, data, nb, order: np.ndarray, K: int = 10) -> np.ndarray:
    """z после удаления первых k рёбер order, k = 0..K (после исчерпания рёбер — константа)."""
    n = int(nb.var.sum())
    keep = np.ones(n, bool)
    out = [z_keep(model, data, nb, keep)]
    for k in range(1, K + 1):
        if k <= n:
            keep[order[k - 1]] = False
            out.append(z_keep(model, data, nb, keep))
        else:
            out.append(out[-1])
    return np.array(out)


def aopc(curve: np.ndarray) -> float:
    return float(np.mean(curve[0] - curve[1:]))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def gnnexplainer_mask(model, data, nb, epochs: int = 200, lr: float = 0.01,
                      size_coef: float = 0.005, ent_coef: float = 1.0) -> np.ndarray:
    """Цель GNNExplainer (Ying et al., 2019; коэффициенты по умолчанию PyG) по тем же весам рёбер E_v."""
    x_sub = data.x[nb.nodes]
    n = int(nb.var.sum())
    g = torch.Generator().manual_seed(0)
    theta = (torch.randn(n, generator=g) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        m = torch.sigmoid(theta)
        w = torch.ones(nb.edge_index.size(1))
        w = w.masked_scatter(nb.var, m)
        logits = model(x_sub, nb.edge_index, w)[nb.center]
        ent = -(m * torch.log(m + 1e-15) + (1 - m) * torch.log(1 - m + 1e-15))
        loss = -F.log_softmax(logits, -1)[C.ILLICIT] + size_coef * m.sum() + ent_coef * ent.mean()
        loss.backward()
        opt.step()
    return torch.sigmoid(theta).detach().numpy()


def randomized(model, level: int, seed: int = 123):
    """Каскадная рандомизация (Adebayo et al., 2018): level 1 — выходной слой и skip, level 2 — ещё и conv1."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    m.conv2.reset_parameters()
    m.skip.reset_parameters()
    if level >= 2:
        m.conv1.reset_parameters()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()
