"""Шаг 3 плана: М4 (жадный контрфактуал, M-9), М3 (похожие случаи, M-8)
и разбивка результатов шага 2 по периодам и меткам.

Оценочная выборка алертов: все размеченные + случайные --unknown-sample неразмеченных.

    python scripts/step3_cf_knn.py
    python scripts/step3_cf_knn.py --limit 50
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
from src.counterfactual import greedy_cf  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext  # noqa: E402
from src.knn import embeddings, knn, knn_vote, reference_set  # noqa: E402
from src.model import load_model  # noqa: E402

LABEL = {C.LICIT: "licit", C.ILLICIT: "illicit", C.UNKNOWN: "unknown"}


def period(step):
    return np.where(np.asarray(step) < C.SHUTDOWN_STEP, f"до {C.SHUTDOWN_STEP}", f"с {C.SHUTDOWN_STEP}")


def step2_breakdown(al: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Результаты шага 2 по группам период × метка."""
    al = al.assign(period=period(al.step), label_name=al.label.map(LABEL))
    tot = al.m2_local.abs() + al.m2_aggregated.abs() + al.m2_edges.abs()
    al = al.assign(sh_local=al.m2_local.abs() / tot, sh_agg=al.m2_aggregated.abs() / tot,
                   sh_edges=al.m2_edges.abs() / tot,
                   holds_wo_edges=al.z_no_edges >= np.log(tau / (1 - tau)),
                   small=al.n_edges_var <= 5)
    g = al.groupby(["period", "label_name"])
    return g.agg(n=("v", "size"), med_edges=("n_edges_var", "median"), share_Ev_le5=("small", "mean"),
                 holds_without_edges=("holds_wo_edges", "mean"), m2_local=("sh_local", "median"),
                 m2_agg=("sh_agg", "median"), m2_edges=("sh_edges", "median")).round(3).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unknown-sample", type=int, default=500)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exp_dir = args.out / "explanations" / f"seed{args.seed}"
    data = load_elliptic(args.root)
    model = load_model(args.seed, data.num_features, args.out)
    for p in model.parameters():
        p.requires_grad_(False)
    tau = json.loads((args.out / "thresholds.json").read_text())[f"gcn_seed{args.seed}"]["tau"]
    p_all = np.load(args.out / "scores.npz")[f"gcn_seed{args.seed}"]
    al = pd.read_csv(exp_dir / "alerts.csv")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("== Шаг 2 по группам (все алерты)")
    br = step2_breakdown(al, tau)
    br.to_csv(exp_dir / "step2_breakdown.csv", index=False)
    print(br.to_string(index=False))

    # Оценочная выборка
    rng = np.random.default_rng(0)
    lab = al[al.label != C.UNKNOWN]
    unk = al[al.label == C.UNKNOWN]
    unk = unk.iloc[rng.permutation(len(unk))[: args.unknown_sample]]
    ev = pd.concat([lab, unk]).sort_values("v").reset_index(drop=True)
    if args.limit:
        ev = ev.iloc[: args.limit]
    print(f"\n== Оценочная выборка: {len(ev)} алертов ({len(lab)} размеченных + {len(unk)} неразмеченных)")

    # М4
    print(f"\n== М4: жадный контрфактуал, K = {args.kmax}")
    ctx = GraphContext(data)
    cf_rows, t0 = [], time.time()
    for i, v in enumerate(ev.v.to_numpy()):
        nb = ctx.neighborhood(int(v))
        r = greedy_cf(model, data, nb, tau, k_max=args.kmax)
        cf_rows.append(dict(v=int(v), cf_success=r["success"], cf_size=r["size"],
                            cf_removed=" ".join(map(str, r["removed"])),
                            z_start=r["z_path"][0], z_end=r["z_path"][-1]))
        if (i + 1) % 100 == 0 or i + 1 == len(ev):
            el = time.time() - t0
            print(f"  {i + 1}/{len(ev)} | {el:.0f} c | ~{el / (i + 1) * (len(ev) - i - 1) / 60:.1f} мин осталось",
                  flush=True)
    cf = ev[["v", "tx_id", "step", "label", "p", "n_edges_var", "z_no_edges"]].merge(pd.DataFrame(cf_rows), on="v")
    cf["period"] = period(cf.step)
    cf["label_name"] = cf.label.map(LABEL)
    cf["cf_le3"] = cf.cf_success & (cf.cf_size <= 3)
    cf.to_csv(exp_dir / "cf.csv", index=False)
    m9 = cf.groupby(["period", "label_name"]).agg(
        n=("v", "size"), cf_found=("cf_success", "mean"), cf_le3=("cf_le3", "mean"),
        median_size=("cf_size", "median")).round(3).reset_index()
    m9.to_csv(exp_dir / "m9_summary.csv", index=False)
    print(m9.to_string(index=False))
    print("  распределение |S| (найденные):", cf.cf_size.dropna().astype(int).value_counts().sort_index().to_dict())

    # М3
    print("\n== М3: 5 похожих размеченных транзакций шагов 1–29")
    h = embeddings(model, data)
    ref = reference_set(data)
    q = torch.as_tensor(ev.v.to_numpy().copy())
    nbr, sim = knn(h, q, ref, k=5)
    rows = []
    for qi, v in enumerate(ev.v.to_numpy()):
        for r in range(5):
            u = int(nbr[qi, r])
            rows.append(dict(v=int(v), rank=r + 1, nbr=u, nbr_tx_id=int(data.tx_id[u]),
                             nbr_label=LABEL[int(data.y[u])], nbr_step=int(data.time_step[u]),
                             sim=float(sim[qi, r])))
    pd.DataFrame(rows).to_csv(exp_dir / "knn.csv", index=False)
    purity = knn_vote(data, nbr)  # все алерты — прогноз illicit, поэтому чистота = доля illicit-соседей
    ev_p = ev.assign(purity=purity, period=period(ev.step), label_name=ev.label.map(LABEL))

    # M-8: согласие голосования kNN с GCN и с истиной на всех размеченных тестовых узлах
    test = data.test_mask.nonzero().flatten()
    nbr_t, _ = knn(h, test, ref, k=5)
    vote = knn_vote(data, nbr_t) >= 0.6
    gcn = p_all[test.numpy()] >= tau
    truth = (data.y[test] == C.ILLICIT).numpy()
    per_t = period(data.time_step[test].numpy())
    m8 = []
    for per in np.unique(per_t):
        m = per_t == per
        m8.append(dict(period=per, agree_knn_gcn=float((vote[m] == gcn[m]).mean()),
                       knn_acc=float((vote[m] == truth[m]).mean()), gcn_acc=float((gcn[m] == truth[m]).mean())))
    m8 = pd.DataFrame(m8).round(3)
    pur = ev_p.groupby(["period", "label_name"]).purity.agg(["size", "mean", "median"]).round(3).reset_index()
    pur.to_csv(exp_dir / "m8_purity.csv", index=False)
    m8.to_csv(exp_dir / "m8_agreement.csv", index=False)
    print("  M-8 чистота (доля illicit среди 5 соседей алерта):")
    print(pur.to_string(index=False))
    print("  M-8 согласие голосования kNN (≥3 из 5) с GCN на размеченном тесте:")
    print(m8.to_string(index=False))
    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин. Файлы: {exp_dir}")


if __name__ == "__main__":
    main()
