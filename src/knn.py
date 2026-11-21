"""Метод М3: похожие размеченные случаи в пространстве скрытого представления GCN."""
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


@torch.no_grad()
def embeddings(model, data) -> torch.Tensor:
    return model.embed(data.x, data.edge_index)


def reference_set(data) -> torch.Tensor:
    """Размеченные транзакции обучающего периода (шаги 1–29): без утечки из будущего."""
    return data.train_mask.nonzero().flatten()


@torch.no_grad()
def knn(h: torch.Tensor, queries: torch.Tensor, ref: torch.Tensor, k: int = 5, chunk: int = 2048):
    """Косинусные k ближайших из ref для каждого запроса. Возвращает (индексы узлов [Q,k], сходства [Q,k])."""
    hr = F.normalize(h[ref], dim=1)
    idx, sims = [], []
    for i in range(0, len(queries), chunk):
        hq = F.normalize(h[queries[i:i + chunk]], dim=1)
        s, j = (hq @ hr.T).topk(k, dim=1)
        idx.append(ref[j])
        sims.append(s)
    return torch.cat(idx), torch.cat(sims)


def knn_vote(data, nbr_idx: torch.Tensor) -> np.ndarray:
    """Доля illicit среди k соседей."""
    return (data.y[nbr_idx] == C.ILLICIT).float().mean(dim=1).numpy()
