"""Шаг 4 плана: метрики верности М1.

M-1 Fidelity− и M-2 Fidelity+ (top-5), M-3 кривая удаления / AOPC против случайного порядка
и порядка «по степени», M-4 восстановление внедрённого ребра, M-5 устойчивость
(сиды, число шагов IG, GNNExplainer), M-6 каскадная рандомизация модели.

    python scripts/step4_faithfulness.py
    python scripts/step4_faithfulness.py --limit 100 --n-stab 50 --n-plant 50
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext, ig_edges  # noqa: E402
from src.faithfulness import (aopc, deletion_curve, fidelity, gnnexplainer_mask, jaccard,  # noqa: E402
                              randomized)
from src.model import load_model, log_odds  # noqa: E402

K_TOP, K_CURVE, N_RAND = 5, 10, 5


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def topk(attr, k):
    return np.argsort(-attr, kind="stable")[:k]


def spear(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-stab", type=int, default=300, help="алертов для M-5 и M-6")
    ap.add_argument("--n-plant", type=int, default=200, help="испытаний для M-4")
    args = ap.parse_args()
    t_start = time.time()
    rng = np.random.default_rng(0)

    exp = args.out / "explanations" / f"seed{args.seed}"
    data = load_elliptic(args.root)
    model = load_model(args.seed, data.num_features, args.out)
    for p in model.parameters():
        p.requires_grad_(False)
    taus = json.loads((args.out / "thresholds.json").read_text())
    tau = taus[f"gcn_seed{args.seed}"]["tau"]
    lt = float(np.log(tau / (1 - tau)))
    cf = pd.read_csv(exp / "cf.csv")
    if args.limit:
        cf = cf.iloc[: args.limit]
    ig_all = pd.read_csv(exp / "edge_ig.csv.gz")
    ig_all = ig_all[ig_all.v.isin(cf.v)]
    ig_by_v = {v: dict(zip(g.edge_id, g.ig)) for v, g in ig_all.groupby("v")}

    ctx = GraphContext(data)
    und = ctx.ei_und
    deg = torch.bincount(und[0], minlength=data.num_nodes).numpy()
    pd.set_option("display.width", 160)

    # ---------- M-1, M-2, M-3
    print(f"== M-1…M-3 на {len(cf)} алертах")
    rows, t0 = [], time.time()
    for i, r in enumerate(cf.itertuples()):
        nb = ctx.neighborhood(int(r.v))
        ids = nb.var_edge_ids.numpy()
        n = len(ids)
        attr = np.array([ig_by_v.get(r.v, {}).get(int(e), 0.0) for e in ids])
        o_ig = topk(attr, n)
        src, dst = data.edge_index[:, ids].numpy()
        o_deg = np.argsort(-np.maximum(deg[src], deg[dst]), kind="stable")
        fid = fidelity(model, data, nb, o_ig, K_TOP, lt)
        rand_fid = [fidelity(model, data, nb, rng.permutation(n), K_TOP, lt) for _ in range(N_RAND)]
        a_ig = aopc(deletion_curve(model, data, nb, o_ig, K_CURVE))
        a_deg = aopc(deletion_curve(model, data, nb, o_deg, K_CURVE))
        a_rand = np.mean([aopc(deletion_curve(model, data, nb, rng.permutation(n), K_CURVE)) for _ in range(N_RAND)])
        rows.append(dict(v=r.v, n_edges=n, period=r.period, label=r.label_name,
                         fid_minus_kept=fid["fid_minus_kept"], fid_minus_ratio=fid["fid_minus_ratio"],
                         fid_plus=fid["fid_plus"], fid_plus_flip=fid["fid_plus_flip"],
                         rand_fid_plus=np.mean([f["fid_plus"] for f in rand_fid]),
                         rand_fid_minus_kept=np.mean([f["fid_minus_kept"] for f in rand_fid]),
                         aopc_ig=a_ig, aopc_rand=a_rand, aopc_deg=a_deg))
        if (i + 1) % 200 == 0 or i + 1 == len(cf):
            print(f"  {i + 1}/{len(cf)} | {time.time() - t0:.0f} c", flush=True)
    fm = pd.DataFrame(rows)
    fm.to_csv(exp / "m1_m3_per_alert.csv", index=False)

    def block(df, name):
        out = dict(subset=name, n=len(df),
                   fid_minus_kept=df.fid_minus_kept.mean(), rand_fid_minus_kept=df.rand_fid_minus_kept.mean(),
                   fid_plus_flip=df.fid_plus_flip.mean())
        for col, lab in [("fid_plus", "fid_plus"), ("aopc_ig", "aopc_ig"), ("aopc_rand", "aopc_rand"),
                         ("aopc_deg", "aopc_deg")]:
            out[lab] = df[col].mean()
        m, lo, hi = boot_ci(df.fid_plus - df.rand_fid_plus)
        out["fid_plus_minus_rand_CI"] = f"{m:.2f} [{lo:.2f}; {hi:.2f}]"
        m, lo, hi = boot_ci(df.aopc_ig - df.aopc_rand)
        out["aopc_ig_minus_rand_CI"] = f"{m:.2f} [{lo:.2f}; {hi:.2f}]"
        return out

    subsets = [(fm, "все"), (fm[fm.n_edges > 10], "|E_v| > 10")]
    for per in sorted(fm.period.unique()):
        for lab in ["illicit", "licit"]:
            subsets.append((fm[(fm.period == per) & (fm.label == lab)], f"{per}, {lab}"))
    tab = pd.DataFrame([block(d, s) for d, s in subsets if len(d) > 1]).round(3)
    tab.to_csv(exp / "m1_m3_summary.csv", index=False)
    print(tab.to_string(index=False))

    # ---------- M-4: внедрённое ребро
    print(f"\n== M-4: внедрённое ребро, {args.n_plant} испытаний")
    p_all = np.load(args.out / "scores.npz")[f"gcn_seed{args.seed}"]
    ts = data.time_step.numpy()
    y = data.y.numpy()
    lo, hi = C.TEST_STEPS
    cand = np.where((y == C.LICIT) & (ts >= lo) & (ts <= hi) & (p_all < 0.1))[0]
    cand = rng.permutation(cand)[: args.n_plant]
    pl_rows = []
    for v in cand:
        illicit_same = np.where((y == C.ILLICIT) & (ts == ts[v]))[0]
        if len(illicit_same) == 0:
            continue
        u = int(rng.choice(illicit_same))
        ei2 = torch.cat([data.edge_index, torch.tensor([[u], [int(v)]])], dim=1)
        d2 = SimpleNamespace(x=data.x, edge_index=ei2, num_nodes=data.num_nodes)
        nb = GraphContext(d2).neighborhood(int(v))
        r1 = ig_edges(model, d2, nb)
        nb0 = ctx.neighborhood(int(v))
        with torch.no_grad():
            z_before = float(log_odds(model(data.x[nb0.nodes], nb0.edge_index)[nb0.center]))
        dz = r1["z"] - z_before
        planted = ei2.size(1) - 1
        ids = nb.var_edge_ids.numpy()
        rank = int(np.where(np.argsort(-r1["attr"], kind="stable") == int(np.where(ids == planted)[0][0]))[0][0]) + 1
        pl_rows.append(dict(v=int(v), u=u, step=int(ts[v]), dz=dz, rank=rank, n_edges=len(ids)))
    pl = pd.DataFrame(pl_rows)
    pl.to_csv(exp / "m4_planted.csv", index=False)
    eff = pl[pl.dz >= 1]
    m4 = dict(trials=len(pl), effective_dz_ge_1=len(eff), share_effective=len(eff) / max(len(pl), 1),
              recall_at_1=float((eff["rank"] <= 1).mean()) if len(eff) else np.nan,
              recall_at_3=float((eff["rank"] <= 3).mean()) if len(eff) else np.nan,
              median_edges=float(eff.n_edges.median()) if len(eff) else np.nan)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in m4.items()})

    # ---------- M-5 и M-6 на подвыборке
    st = cf.iloc[rng.permutation(len(cf))[: args.n_stab]]
    print(f"\n== M-5, M-6 на {len(st)} алертах")
    others = [s for s in C.SEEDS if s != args.seed and (args.out / "models" / f"skipgcn_seed{s}.pt").exists()]
    seed_models = {s: load_model(s, data.num_features, args.out) for s in others}
    for m in seed_models.values():
        for p in m.parameters():
            p.requires_grad_(False)
    rand1, rand2 = randomized(model, 1), randomized(model, 2)
    st_rows, t0 = [], time.time()
    for i, r in enumerate(st.itertuples()):
        nb = ctx.neighborhood(int(r.v))
        ids = nb.var_edge_ids.numpy()
        a0 = np.array([ig_by_v.get(r.v, {}).get(int(e), 0.0) for e in ids])
        tops = [topk(a0, K_TOP)]
        for m in seed_models.values():
            tops.append(topk(ig_edges(m, data, nb)["attr"], K_TOP))
        pairs = [jaccard(ids[a], ids[b]) for i_, a in enumerate(tops) for b in tops[i_ + 1:]]
        a256 = ig_edges(model, data, nb, n_steps=256, max_steps=256)["attr"]
        gx = gnnexplainer_mask(model, data, nb)
        ar1 = ig_edges(rand1, data, nb, max_steps=64)["attr"]
        ar2 = ig_edges(rand2, data, nb, max_steps=64)["attr"]
        st_rows.append(dict(v=r.v, n_edges=len(ids), period=r.period, label=r.label_name,
                            jaccard_seeds=np.mean(pairs) if pairs else np.nan,
                            spearman_64_256=spear(a0, a256), spearman_gnnexplainer=spear(a0, gx),
                            jaccard_gnnexplainer=jaccard(ids[topk(a0, K_TOP)], ids[topk(gx, K_TOP)]),
                            spearman_rand_top=spear(a0, ar1), spearman_rand_all=spear(a0, ar2)))
        if (i + 1) % 50 == 0 or i + 1 == len(st):
            print(f"  {i + 1}/{len(st)} | {time.time() - t0:.0f} c", flush=True)
    sd = pd.DataFrame(st_rows)
    sd.to_csv(exp / "m5_m6_per_alert.csv", index=False)
    rows = []
    for d, name in [(sd, "все"), (sd[sd.n_edges > 10], "|E_v| > 10")]:
        rows.append(dict(subset=name, n=len(d), jaccard_seeds_median=d.jaccard_seeds.median(),
                         spearman_64_256_median=d.spearman_64_256.median(),
                         spearman_gnnexpl_median=d.spearman_gnnexplainer.median(),
                         jaccard_gnnexpl_median=d.jaccard_gnnexplainer.median(),
                         spearman_rand_top_median=d.spearman_rand_top.median(),
                         spearman_rand_all_median=d.spearman_rand_all.median()))
    m56 = pd.DataFrame(rows).round(3)
    m56.to_csv(exp / "m5_m6_summary.csv", index=False)
    print(m56.to_string(index=False))

    (exp / "m4_summary.json").write_text(json.dumps(m4, ensure_ascii=False, indent=2))
    print(f"\nГотово за {(time.time() - t_start) / 60:.1f} мин. Файлы: {exp}")


if __name__ == "__main__":
    main()
