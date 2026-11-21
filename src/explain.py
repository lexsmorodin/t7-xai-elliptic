"""Методы М1 и М2 дизайн-документа: Integrated Gradients по весам рёбер и по признакам.

Окрестность. Двухслойная GCN получает информацию только из 2-хоп окрестности v,
но нормировка D^{-1/2} зависит от степеней узлов на расстоянии 2, а их степени — от рёбер
к узлам на расстоянии 3. Поэтому считаем на индуцированном 3-хоп подграфе (тогда z(v)
совпадает с z на всём графе), а переменными объявляем только рёбра E_v, оба конца которых
лежат в 2-хоп окрестности. Остальные рёбра подграфа зафиксированы с весом 1.
"""
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.utils import k_hop_subgraph

from . import config as C
from .model import log_odds

N_LOCAL = 93  # первые 93 признака PyG — локальные, последние 72 — агрегаты по соседям


@dataclass
class Neighborhood:
    v: int                  # глобальный индекс узла
    center: int             # индекс v в подграфе
    nodes: torch.Tensor     # глобальные индексы узлов подграфа
    edge_index: torch.Tensor  # локальные индексы направленных рёбер подграфа [2, E_sub]
    edge_ids: torch.Tensor  # глобальные id этих рёбер (столбцы data.edge_index)
    var: torch.Tensor       # bool [E_sub]: ребро входит в E_v (переменная для IG)

    @property
    def var_edge_ids(self):
        return self.edge_ids[self.var]


class GraphContext:
    """Кеш неориентированного графа для быстрого построения окрестностей."""

    def __init__(self, data):
        self.data = data
        ei = data.edge_index
        self.ei_und = torch.cat([ei, ei.flip(0)], dim=1)
        self.n = data.num_nodes

    def neighborhood(self, v: int, hops: int = 2) -> Neighborhood:
        ei = self.data.edge_index
        far, _, _, _ = k_hop_subgraph(v, hops + 1, self.ei_und, num_nodes=self.n)
        near, _, _, _ = k_hop_subgraph(v, hops, self.ei_und, num_nodes=self.n)
        in_far = torch.zeros(self.n, dtype=torch.bool)
        in_far[far] = True
        in_near = torch.zeros(self.n, dtype=torch.bool)
        in_near[near] = True
        src, dst = ei
        keep = in_far[src] & in_far[dst]
        edge_ids = keep.nonzero().flatten()
        remap = torch.full((self.n,), -1, dtype=torch.long)
        remap[far] = torch.arange(far.numel())
        local = remap[ei[:, edge_ids]]
        var = in_near[src[edge_ids]] & in_near[dst[edge_ids]]
        return Neighborhood(v, int(remap[v]), far, local, edge_ids, var)


def _z_batch(model, x_sub, nb: Neighborhood, w_var: torch.Tensor, x_center: torch.Tensor | None = None):
    """z(v) для пакета весов w_var [B, |E_v|] (и, если задано, признаков центра [B, 165])."""
    out = []
    for b in range(w_var.shape[0]):
        w = torch.ones(nb.edge_index.size(1), dtype=w_var.dtype)
        w = w.masked_scatter(nb.var, w_var[b])
        x = x_sub
        if x_center is not None:
            x = x_sub.clone()
            x[nb.center] = x_center[b]
        out.append(log_odds(model(x, nb.edge_index, w)[nb.center]))
    return torch.stack(out)


def ig_edges(model, data, nb: Neighborhood, n_steps: int = 64, max_steps: int = 512, tol: float = 0.05):
    """М1. IG по весам рёбер E_v, baseline w = 0, признаки фиксированы.
    Если ошибка полноты > tol·|Δz|, число шагов удваивается (до max_steps)."""
    x_sub = data.x[nb.nodes]
    n_var = int(nb.var.sum())
    ones, zeros = torch.ones(1, n_var), torch.zeros(1, n_var)
    with torch.no_grad():
        z1 = float(_z_batch(model, x_sub, nb, ones))
        z0 = float(_z_batch(model, x_sub, nb, zeros))
    dz = z1 - z0
    if n_var == 0:
        return dict(attr=np.zeros(0), z=z1, z_no_edges=z0, delta=0.0, rel_err=0.0, n_steps=0)
    ig = IntegratedGradients(lambda w: _z_batch(model, x_sub, nb, w))
    steps = n_steps
    while True:
        attr, delta = ig.attribute(ones, baselines=zeros, n_steps=steps, return_convergence_delta=True)
        rel = abs(float(delta)) / max(abs(dz), 1e-8)
        if rel <= tol or steps >= max_steps:
            break
        steps *= 2
    return dict(attr=attr.detach().numpy()[0], z=z1, z_no_edges=z0, delta=float(delta), rel_err=rel, n_steps=steps)


def licit_baseline(data) -> torch.Tensor:
    """x' — медиана признаков licit-транзакций обучающего периода (шаги 1–29)."""
    m = data.train_mask & (data.y == C.LICIT)
    return data.x[m].median(dim=0).values


def ig_joint(model, data, nb: Neighborhood, x_base: torch.Tensor, n_steps: int = 64,
             max_steps: int = 512, tol: float = 0.05):
    """М2. Один IG по паре (признаки v, веса E_v) от (x', 0) к (x_v, 1).
    Возвращает три суммы: локальные признаки, агрегаты, рёбра."""
    x_sub = data.x[nb.nodes]
    n_var = int(nb.var.sum())
    xv = data.x[nb.v].unsqueeze(0)
    xb = x_base.unsqueeze(0)
    ones, zeros = torch.ones(1, n_var), torch.zeros(1, n_var)

    def f(xc, w):
        return _z_batch(model, x_sub, nb, w, xc)

    with torch.no_grad():
        z1, zb = float(f(xv, ones)), float(f(xb, zeros))
    ig = IntegratedGradients(f)
    steps = n_steps
    while True:
        (a_x, a_w), delta = ig.attribute((xv, ones), baselines=(xb, zeros), n_steps=steps,
                                         return_convergence_delta=True)
        rel = abs(float(delta)) / max(abs(z1 - zb), 1e-8)
        if rel <= tol or steps >= max_steps:
            break
        steps *= 2
    a_x = a_x.detach().numpy()[0]
    return dict(local=float(a_x[:N_LOCAL].sum()), aggregated=float(a_x[N_LOCAL:].sum()),
                edges=float(a_w.sum()), z=z1, z_base=zb, delta=float(delta), rel_err=rel, n_steps=steps)
