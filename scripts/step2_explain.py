"""Шаг 2 плана: методы М1 и М2 для всех алертов на тесте, метрика полноты M-7.

    python scripts/step2_explain.py                 # модель seed 0, все алерты шагов 35–49
    python scripts/step2_explain.py --limit 50      # быстрая проверка
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext, ig_edges, ig_joint, licit_baseline  # noqa: E402
from src.model import load_model, log_odds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64, help="начальное число шагов IG")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N алертов (0 = все)")
    args = ap.parse_args()

    data = load_elliptic(args.root)
    model = load_model(args.seed, data.num_features, args.out)
    for p in model.parameters():
        p.requires_grad_(False)
    tau = json.loads((args.out / "thresholds.json").read_text())[f"gcn_seed{args.seed}"]["tau"]
    p_all = np.load(args.out / "scores.npz")[f"gcn_seed{args.seed}"]

    ts = data.time_step.numpy()
    lo, hi = C.TEST_STEPS
    alerts = np.where((p_all >= tau) & (ts >= lo) & (ts <= hi))[0]
    if args.limit:
        alerts = alerts[: args.limit]
    print(f"seed {args.seed}: τ = {tau:.4f}, алертов на шагах {lo}–{hi}: {len(alerts)}")

    with torch.no_grad():
        z_full = log_odds(model(data.x, data.edge_index)).numpy()
    ctx = GraphContext(data)
    x_base = licit_baseline(data)
    rows, edge_rows = [], []
    t0 = time.time()
    for i, v in enumerate(alerts):
        v = int(v)
        nb = ctx.neighborhood(v)
        m1 = ig_edges(model, data, nb, n_steps=args.steps)
        m2 = ig_joint(model, data, nb, x_base, n_steps=args.steps)
        ids = nb.var_edge_ids.numpy()
        src, dst = data.edge_index[:, ids].numpy()
        for e, s, d, a in zip(ids, src, dst, m1["attr"]):
            edge_rows.append((v, int(e), int(s), int(d), float(a)))
        rows.append(dict(
            v=v, tx_id=int(data.tx_id[v]), step=int(ts[v]), label=int(data.y[v]), p=float(p_all[v]),
            n_nodes_3hop=int(nb.nodes.numel()), n_edges_var=len(ids),
            z=m1["z"], z_no_edges=m1["z_no_edges"], m1_rel_err=m1["rel_err"], m1_steps=m1["n_steps"],
            z_base=m2["z_base"], m2_local=m2["local"], m2_aggregated=m2["aggregated"], m2_edges=m2["edges"],
            m2_rel_err=m2["rel_err"], m2_steps=m2["n_steps"],
            z_full_check=abs(m1["z"] - float(z_full[v])),
        ))
        if (i + 1) % 50 == 0 or i + 1 == len(alerts):
            el = time.time() - t0
            print(f"  {i + 1}/{len(alerts)} | {el:.0f} c | ~{el / (i + 1) * (len(alerts) - i - 1) / 60:.1f} мин осталось",
                  flush=True)

    out = args.out / "explanations" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "alerts.csv", index=False)
    pd.DataFrame(edge_rows, columns=["v", "edge_id", "src", "dst", "ig"]).to_csv(out / "edge_ig.csv.gz", index=False)

    # Сводка: M-7 и разложение М2
    total = df.z - df.z_base
    share = lambda col: (df[col].abs() / (df.m2_local.abs() + df.m2_aggregated.abs() + df.m2_edges.abs())).median()
    summary = {
        "алертов": len(df),
        "z на подграфе = z на графе (max |разница|)": float(df.z_full_check.max()),
        "M-7 М1: медиана ошибки полноты": float(df.m1_rel_err.median()),
        "M-7 М1: доля > 5%": float((df.m1_rel_err > 0.05).mean()),
        "M-7 М2: медиана ошибки полноты": float(df.m2_rel_err.median()),
        "M-7 М2: доля > 5%": float((df.m2_rel_err > 0.05).mean()),
        "медиана |E_v|": float(df.n_edges_var.median()),
        "доля алертов без рёбер в окрестности": float((df.n_edges_var == 0).mean()),
        "М2: медианная доля |локальные|": float(share("m2_local")),
        "М2: медианная доля |агрегаты|": float(share("m2_aggregated")),
        "М2: медианная доля |рёбра|": float(share("m2_edges")),
        "доля алертов, где флаг держится без рёбер (z_no_edges ≥ logit τ)":
            float((df.z_no_edges >= np.log(tau / (1 - tau))).mean()),
        "медиана z − z_base": float(total.median()),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Файлы: {out}")


if __name__ == "__main__":
    main()
