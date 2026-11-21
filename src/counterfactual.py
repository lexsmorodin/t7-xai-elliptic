"""Метод М4: жадный адаптивный минимальный контрфактуал по рёбрам E_v."""
import numpy as np
import torch
from captum.attr import IntegratedGradients

from .explain import Neighborhood, _z_batch


def greedy_cf(model, data, nb: Neighborhood, tau: float, k_max: int = 10, n_steps: int = 64):
    """На каждом шаге пересчитываем IG по рёбрам на текущем графе (baseline w = 0)
    и обнуляем ребро с наибольшим вкладом. Стоп, когда p_illicit < τ или после k_max удалений.

    Возвращает удалённые рёбра (глобальные id) по порядку, траекторию z и флаг успеха.
    Жадный поиск даёт верхнюю оценку минимального размера.
    """
    x_sub = data.x[nb.nodes]
    n_var = int(nb.var.sum())
    logit_tau = float(np.log(tau / (1 - tau)))
    var_ids = nb.var_edge_ids.numpy()
    w = torch.ones(1, n_var)
    with torch.no_grad():
        z = float(_z_batch(model, x_sub, nb, w))
    path, removed = [z], []
    ig = IntegratedGradients(lambda ww: _z_batch(model, x_sub, nb, ww))
    while z >= logit_tau and len(removed) < min(k_max, n_var):
        attr = ig.attribute(w, baselines=torch.zeros_like(w), n_steps=n_steps)[0].detach()
        attr[w[0] == 0] = -np.inf
        j = int(attr.argmax())
        w[0, j] = 0.0
        removed.append(int(var_ids[j]))
        with torch.no_grad():
            z = float(_z_batch(model, x_sub, nb, w))
        path.append(z)
    return dict(success=z < logit_tau, size=len(removed) if z < logit_tau else None,
                removed=removed, z_path=path)
