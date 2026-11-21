"""Skip-GCN (Weber et al., 2019) с весами рёбер как дифференцируемым входом.

Объект объяснения — направленный поток e из edge_index. Внутри модели граф
симметризуется: оба направления потока получают один и тот же вес w_e, поэтому
IG по w (метод М1) даёт один вклад на один реальный поток средств.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from . import config as C


def symmetrize(edge_index: torch.Tensor, edge_weight: torch.Tensor):
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    w = torch.cat([edge_weight, edge_weight])
    return ei, w


class SkipGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = C.HIDDEN, out_dim: int = 2):
        super().__init__()
        # normalize=True, add_self_loops=True: Â = D^{-1/2}(A_w + I)D^{-1/2}, нормировка
        # пересчитывается при каждом новом w (раздел 2 «Доступ»)
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def _graph(self, edge_index, edge_weight, x):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
        return symmetrize(edge_index, edge_weight)

    def embed(self, x, edge_index, edge_weight=None):
        """Скрытое представление h_v после первого слоя (метод М3)."""
        ei, w = self._graph(edge_index, edge_weight, x)
        return F.relu(self.conv1(x, ei, w))

    def forward(self, x, edge_index, edge_weight=None):
        ei, w = self._graph(edge_index, edge_weight, x)
        h = F.relu(self.conv1(x, ei, w))
        return self.conv2(h, ei, w) + self.skip(x)


def log_odds(logits: torch.Tensor) -> torch.Tensor:
    """z = log p_illicit − log p_licit (объясняемая величина)."""
    return logits[..., C.ILLICIT] - logits[..., C.LICIT]


def p_illicit(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)[..., C.ILLICIT]


def model_path(seed: int, out: Path = C.ARTIFACTS) -> Path:
    return Path(out) / "models" / f"skipgcn_seed{seed}.pt"


def load_model(seed: int, in_dim: int = 165, out: Path = C.ARTIFACTS) -> SkipGCN:
    m = SkipGCN(in_dim)
    m.load_state_dict(torch.load(model_path(seed, out), map_location="cpu"))
    return m.eval()
