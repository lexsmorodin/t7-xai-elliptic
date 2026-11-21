"""Шаг 6 плана: хрупкость (раздел Б документа).

Б.2 Камуфляж: к верным алертам (illicit, до шага 43) добавляем m = 1, 3, 5 потоков от случайных
    licit-транзакций того же шага. Смотрим долю снятых флагов и знак вклада IG добавленных рёбер.
Б.3 Подбор модели банком: банк может выбрать любой из 5 сидов (модели одинаковой точности).
    Сравниваем долю «годных обоснований» М5 (строгий C5) при фиксированном seed 0
    и при выборе «лучшего» сида для каждого алерта, а также precision этих наборов.
Б.1 (сдвиг) и Б.4 (шум меток) уже покрыты: M-10 и сравнение C5 doc / strict на шаге 5.

    python scripts/step6_fragility.py
    python scripts/step6_fragility.py --n-camo 50 --n-shop 50
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.counterfactual import greedy_cf  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext, ig_edges  # noqa: E402
from src.faithfulness import fidelity  # noqa: E402
from src.model import load_model, log_odds  # noqa: E402
from src.protocol import leads_to, wilson  # noqa: E402


def frozen(seed, data, out):
    m = load_model(seed, data.num_features, out)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--n-camo", type=int, default=200)
    ap.add_argument("--n-shop", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    exp = args.out / "explanations" / "seed0"
    (args.out / "fragility").mkdir(exist_ok=True)
    data = load_elliptic(args.root)
    y, ts = data.y.numpy(), data.time_step.numpy()
    th = json.loads((args.out / "thresholds.json").read_text())
    scores = np.load(args.out / "scores.npz")
    seeds = [s for s in C.SEEDS if f"gcn_seed{s}" in th]
    models = {s: frozen(s, data, args.out) for s in seeds}
    tau = {s: th[f"gcn_seed{s}"]["tau"] for s in seeds}
    lt = {s: float(np.log(tau[s] / (1 - tau[s]))) for s in seeds}
    ctx = GraphContext(data)
    pd.set_option("display.width", 160)

    # ---------- Б.2 Камуфляж
    m0 = models[0]
    tp = np.where((y == C.ILLICIT) & (scores["gcn_seed0"] >= tau[0]) & (ts >= C.TEST_STEPS[0])
                  & (ts < C.SHUTDOWN_STEP))[0]
    tp = rng.permutation(tp)[: args.n_camo]
    print(f"== Б.2 Камуфляж: {len(tp)} верных алертов, m = 1, 3, 5 потоков от licit")
    rows = []
    for v in tp:
        licit_same = np.where((y == C.LICIT) & (ts == ts[v]))[0]
        nb0 = ctx.neighborhood(int(v))
        with torch.no_grad():
            z0 = float(log_odds(m0(data.x[nb0.nodes], nb0.edge_index)[nb0.center]))
        for m in (1, 3, 5):
            src = rng.choice(licit_same, m, replace=False)
            new = torch.tensor(np.stack([src, np.full(m, v)]), dtype=torch.long)
            ei2 = torch.cat([data.edge_index, new], dim=1)
            d2 = SimpleNamespace(x=data.x, edge_index=ei2, num_nodes=data.num_nodes)
            nb = GraphContext(d2).neighborhood(int(v))
            r = ig_edges(m0, d2, nb)
            ids = nb.var_edge_ids.numpy()
            added = ids >= data.edge_index.size(1)
            rows.append(dict(v=int(v), m=m, z_before=z0, z_after=r["z"], flag_removed=r["z"] < lt[0],
                             ig_added_mean=float(r["attr"][added].mean()),
                             ig_added_negative=float((r["attr"][added] < 0).mean())))
    camo = pd.DataFrame(rows)
    camo.to_csv(args.out / "fragility" / "camouflage.csv", index=False)
    camo_s = camo.groupby("m").agg(flag_removed=("flag_removed", "mean"),
                                   dz_mean=("z_after", "mean"), ig_added_mean=("ig_added_mean", "mean"),
                                   ig_added_negative=("ig_added_negative", "mean")).round(3)
    camo_s["dz_mean"] = (camo.groupby("m").z_after.mean() - camo.groupby("m").z_before.mean()).round(3)
    print(camo_s.reset_index().to_string(index=False))

    # ---------- Б.3 Подбор модели банком
    prot = pd.read_csv(exp / "m5_protocol.csv")
    prot = prot[prot.label != C.UNKNOWN]
    shop = prot.iloc[rng.permutation(len(prot))[: args.n_shop]]
    print(f"\n== Б.3 Подбор сида: {len(shop)} размеченных алертов, сиды {seeds}")
    flag_all = {s: scores[f"gcn_seed{s}"] >= tau[s] for s in seeds}
    lab = y != C.UNKNOWN
    prec_step = {s: {t: ((y == C.ILLICIT) & flag_all[s] & lab & (ts == t)).sum()
                     / max((flag_all[s] & lab & (ts == t)).sum(), 1) for t in np.unique(ts)} for s in seeds}
    stab = dict(zip(prot.v, prot.C4))  # устойчивость по сидам не зависит от выбора основного сида
    und = ctx.ei_und
    ei = data.edge_index.numpy()

    ok_strict = y == C.ILLICIT

    rows, t1 = [], time.time()
    for i, r in enumerate(shop.itertuples()):
        v = int(r.v)
        nb = ctx.neighborhood(v)
        for s in seeds:
            if not flag_all[s][v]:
                rows.append(dict(v=v, seed=s, alert=False, valid=False, label=r.label))
                continue
            att = ig_edges(models[s], data, nb)["attr"]
            order = np.argsort(-att, kind="stable")
            c3 = fidelity(models[s], data, nb, order, 5, lt[s])["fid_minus_kept"]
            cf = greedy_cf(models[s], data, nb, tau[s])
            c2 = cf["success"] and cf["size"] <= 3
            c1 = prec_step[s][ts[v]] >= 0.7
            valid = bool(c1 and c2 and c3 and stab[v] and leads_to(v, cf["removed"], ok_strict, ei, und, data.num_nodes))
            rows.append(dict(v=v, seed=s, alert=True, valid=valid, label=r.label))
        if (i + 1) % 50 == 0 or i + 1 == len(shop):
            print(f"  {i + 1}/{len(shop)} | {time.time() - t1:.0f} c", flush=True)
    sh = pd.DataFrame(rows)
    sh.to_csv(args.out / "fragility" / "seed_shopping.csv", index=False)
    per_v = sh.groupby("v").agg(label=("label", "first"), valid_seed0=("valid", lambda s: bool(s.iloc[0])),
                                valid_any=("valid", "any"), n_valid=("valid", "sum"))
    out = []
    for name, col in [("seed 0 (зафиксирован регулятором)", "valid_seed0"), ("лучший из 5 сидов (выбор банка)", "valid_any")]:
        sub = per_v[per_v[col]]
        pr, lo, hi = wilson(int((sub.label == C.ILLICIT).sum()), len(sub))
        dl, dlo, dhi = wilson(len(sub), len(per_v))
        out.append(dict(вариант=name, годных=len(sub), доля=round(dl, 3), доля_ci=f"[{dlo:.3f}; {dhi:.3f}]",
                        precision=round(pr, 3), precision_ci=f"[{lo:.3f}; {hi:.3f}]"))
    flip = per_v[per_v.valid_any != per_v.valid_seed0]
    out_df = pd.DataFrame(out)
    out_df.to_csv(args.out / "fragility" / "seed_shopping_summary.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"  вердикт зависит от выбора сида у {len(flip) / len(per_v):.1%} алертов; "
          f"распределение числа сидов с годным обоснованием: {per_v.n_valid.value_counts().sort_index().to_dict()}")
    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин. Файлы: {args.out / 'fragility'}")


if __name__ == "__main__":
    main()
