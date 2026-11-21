"""Шаг 7: протокол регулятора v2, метрика M-11 с интервалами Уилсона и проверки по рецензии.

Протокол v2 (docs/design_doc_v2.md, М5):
  C1 precision размеченных алертов шага >= 0,7 (ретроспективно)
  C2 контрфактуал М4 найден, |S| <= 3
  C3 каждый поток из S за <= 2 хопа (не через сам алерт) ведёт к транзакции с меткой illicit
  C4 условия C1–C3 выполнены у >= 3 из 5 сидов; основная модель (seed 0) фиксирована регулятором
  Красный флаг: C2 выполнено, C3 нет. Он использует метки того же шага, поэтому только ретроспективный.
  Вычислимый в момент решения вариант — сам C2 (без меток).

Дополнительно:
  M-11 с согласованными базами: каждая группа сравнивается с базой того же слоя (те же шаги);
  M-8 среди алертов; абляция «та же GCN без рёбер»; ложные срабатывания сигнала камуфляжа;
  бутстрап-интервал Жаккара (M-5); доля графа в М2 по шагам (Б.1).

    python scripts/step7_protocol_v2.py
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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.counterfactual import greedy_cf  # noqa: E402
from src.data import load_elliptic  # noqa: E402
from src.explain import GraphContext  # noqa: E402
from src.model import load_model, log_odds  # noqa: E402
from src.protocol import leads_to, wilson  # noqa: E402

PREC_MIN, CF_MAX, SEEDS_MIN = 0.7, 3, 3


def ci_str(k, n):
    p, lo, hi = wilson(k, n)
    return (round(p, 4), f"[{lo:.3f}; {hi:.3f}]") if n else (np.nan, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS)
    args = ap.parse_args()
    t0 = time.time()
    exp = args.out / "explanations" / "seed0"
    data = load_elliptic(args.root)
    y, ts = data.y.numpy(), data.time_step.numpy()
    th = json.loads((args.out / "thresholds.json").read_text())
    scores = np.load(args.out / "scores.npz")
    seeds = [s for s in C.SEEDS if f"gcn_seed{s}" in th]
    tau = {s: th[f"gcn_seed{s}"]["tau"] for s in seeds}
    lt = {s: float(np.log(tau[s] / (1 - tau[s]))) for s in seeds}
    flag = {s: scores[f"gcn_seed{s}"] >= tau[s] for s in seeds}
    lab = y != C.UNKNOWN
    prec_step = {s: {t: ((y == C.ILLICIT) & flag[s] & lab & (ts == t)).sum() / max((flag[s] & lab & (ts == t)).sum(), 1)
                     for t in np.unique(ts)} for s in seeds}
    models = {}
    for s in seeds:
        m = load_model(s, data.num_features, args.out)
        for p in m.parameters():
            p.requires_grad_(False)
        models[s] = m
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    prot = pd.read_csv(exp / "m5_protocol.csv")
    prot["v2_seed0"] = prot.C1 & prot.C2 & prot.C5_strict
    prot["red_flag"] = prot.C2 & ~prot.C5_strict
    prot["holds"] = prot.z_no_edges >= lt[0]
    knn = pd.read_csv(exp / "knn.csv")
    pur = knn.groupby("v").nbr_label.apply(lambda s: (s == "illicit").mean()).rename("knn_purity")
    prot = prot.merge(pur, on="v", how="left")
    prot["knn_major"] = prot.knn_purity >= 0.6

    # ---------- C4 v2: остальные сиды, только для алертов, прошедших C1–C3 у seed 0
    ctx = GraphContext(data)
    und, ei = ctx.ei_und, data.edge_index.numpy()
    ok_strict = y == C.ILLICIT
    cand = prot[prot.v2_seed0]
    print(f"== C4 v2: {len(cand)} алертов прошли C1–C3 у seed 0; считаю сиды {[s for s in seeds if s != 0]}")
    n_ok = {}
    for i, v in enumerate(cand.v.astype(int)):
        nb = ctx.neighborhood(v)
        ok = 1
        for s in seeds:
            if s == 0 or not flag[s][v] or prec_step[s][ts[v]] < PREC_MIN:
                continue
            cf = greedy_cf(models[s], data, nb, tau[s])
            if cf["success"] and cf["size"] <= CF_MAX and leads_to(v, cf["removed"], ok_strict, ei, und, data.num_nodes):
                ok += 1
        n_ok[v] = ok
        if (i + 1) % 25 == 0 or i + 1 == len(cand):
            print(f"  {i + 1}/{len(cand)} | {time.time() - t0:.0f} c", flush=True)
    prot["seeds_ok"] = prot.v.map(n_ok).fillna(0).astype(int)
    prot["v2_valid"] = prot.v2_seed0 & (prot.seeds_ok >= SEEDS_MIN)
    prot.to_csv(exp / "m5_protocol_v2.csv", index=False)

    # ---------- M-11 с согласованными базами
    L = prot[prot.label != C.UNKNOWN].copy()
    ill = (L.label == C.ILLICIT).to_numpy()
    c1_steps = sorted(int(t) for t in L.step.unique() if prec_step[0][t] >= PREC_MIN)
    layers = {"35–49": np.ones(len(L), bool), "35–42": L.period.str.startswith("до").to_numpy(),
              f"шаги C1 {c1_steps}": L.step.isin(c1_steps).to_numpy()}
    lay_all, lay_pre, lay_c1 = list(layers)
    groups = [
        ("все алерты слоя (база)", np.ones(len(L), bool), [lay_all, lay_pre, lay_c1]),
        ("C2: контрфактуал ≤ 3 потоков (вычислим без меток)", L.C2.to_numpy(), [lay_all, lay_pre, lay_c1]),
        ("красный флаг: C2 и не C3 (ретроспективно)", L.red_flag.to_numpy(), [lay_all, lay_pre, lay_c1]),
        ("годные по протоколу v1", L.valid_doc.to_numpy(), [lay_c1]),
        ("годные v1 со строгим C5", L.valid_strict.to_numpy(), [lay_c1]),
        ("v2: C1–C3 у seed 0", L.v2_seed0.to_numpy(), [lay_c1]),
        ("v2: годные (C1–C3 у ≥ 3 из 5 сидов)", L.v2_valid.to_numpy(), [lay_c1]),
        ("флаг держится без потоков", L.holds.to_numpy(), [lay_all, lay_pre, lay_c1]),
        ("М6 A: без потоков ✓, прецеденты ✓", (L.holds & L.knn_major).to_numpy(), [lay_pre]),
        ("М6 B: без потоков ✓, прецеденты ✗", (L.holds & ~L.knn_major).to_numpy(), [lay_pre]),
        ("М6 C: без потоков ✗, прецеденты ✓", (~L.holds & L.knn_major).to_numpy(), [lay_pre]),
        ("М6 D: без потоков ✗, прецеденты ✗", (~L.holds & ~L.knn_major).to_numpy(), [lay_pre]),
    ]
    rows = []
    for name, m, lays in groups:
        for lay in lays:
            base = layers[lay]
            sub = m & base
            p, ci = ci_str(int(ill[sub].sum()), int(sub.sum()))
            bp, bci = ci_str(int(ill[base].sum()), int(base.sum()))
            rows.append(dict(группа=name, слой=lay, n=int(sub.sum()), доля_слоя=round(sub.sum() / base.sum(), 3),
                             precision=p, ci=ci, база=bp, база_ci=bci))
    m11 = pd.DataFrame(rows)
    m11.to_csv(exp / "m11_precision_ci.csv", index=False)
    print("\n== M-11: precision групп размеченных алертов и база того же слоя, 95% интервалы Уилсона")
    print(m11.to_string(index=False))
    print(f"  число сидов с C1–C3 среди прошедших seed 0: "
          f"{prot.loc[prot.v2_seed0, 'seeds_ok'].value_counts().sort_index().to_dict()}")

    # ---------- M-8 среди алертов
    rows = []
    for name, m in [("все размеченные алерты", np.ones(len(L), bool)), ("верные (illicit)", ill), ("ложные (licit)", ~ill)]:
        p, ci = ci_str(int(L.knn_major.to_numpy()[m].sum()), int(m.sum()))
        rows.append(dict(группа=name, n=int(m.sum()), доля_голосования_illicit=p, ci=ci,
                         средняя_чистота=round(float(L.knn_purity[m].mean()), 3)))
    m8 = pd.DataFrame(rows)
    m8.to_csv(exp / "m8_alerts.csv", index=False)
    print("\n== M-8 среди алертов: доля, где большинство из 5 прецедентов illicit (= согласие kNN с GCN)")
    print(m8.to_string(index=False))

    # ---------- Абляция: та же GCN без рёбер; сигнал камуфляжа
    lo, hi = C.TEST_STEPS
    test_lab = lab & (ts >= lo) & (ts <= hi)
    pre = ts < C.SHUTDOWN_STEP
    yb = (y == C.ILLICIT).astype(int)
    abl, z0_all = [], {}
    for s in seeds:
        with torch.no_grad():
            lg0 = models[s](data.x, data.edge_index, torch.zeros(data.num_edges))
        z0 = log_odds(lg0).numpy()
        z0_all[s] = z0
        p0 = 1 / (1 + np.exp(-z0))
        for per, m in [("35–42", test_lab & pre), ("35–49", test_lab)]:
            for name, p in [("с рёбрами", scores[f"gcn_seed{s}"]), ("без рёбер", p0)]:
                pr, rc, f1, _ = precision_recall_fscore_support(yb[m], (p[m] >= tau[s]).astype(int),
                                                               average="binary", zero_division=0)
                abl.append(dict(seed=s, период=per, модель=name, pr_auc=average_precision_score(yb[m], p[m]),
                                precision=pr, recall=rc, f1=f1))
    abl = pd.DataFrame(abl)
    abl_s = abl.groupby(["период", "модель"])[["pr_auc", "precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
    abl.to_csv(exp / "ablation_no_edges.csv", index=False)
    print("\n== Абляция: Skip-GCN с рёбрами и та же модель с нулевыми весами всех рёбер (5 сидов)")
    print(abl_s.to_string())

    with torch.no_grad():
        z_full = log_odds(models[0](data.x, data.edge_index)).numpy()
    sup = (z0_all[0] >= lt[0]) & (z_full < lt[0])  # без рёбер был бы флаг, с рёбрами — нет
    rows = []
    for per, m in [("35–42", test_lab & pre), ("35–49", test_lab)]:
        nonal = m & ~flag[0]
        k_sig, n_sig = int((sup & nonal & (y == C.ILLICIT)).sum()), int((sup & nonal).sum())
        k_base, n_base = int((nonal & (y == C.ILLICIT)).sum()), int(nonal.sum())
        p, ci = ci_str(k_sig, n_sig)
        bp, bci = ci_str(k_base, n_base)
        rows.append(dict(период=per, неалертов=n_base, сигнал=n_sig, доля_сигнала=round(n_sig / n_base, 4),
                         illicit_среди_сигнала=p, ci=ci, illicit_база=bp, база_ci=bci,
                         поймано_illicit=f"{k_sig} из {k_base}"))
    camo = pd.DataFrame(rows)
    camo.to_csv(args.out / "fragility" / "camouflage_signal.csv", index=False)
    print("\n== Сигнал камуфляжа на размеченных транзакциях без флага: «без рёбер был бы флаг»")
    print(camo.to_string(index=False))

    # ---------- Бутстрап Жаккара (M-5) и доля графа по шагам (Б.1)
    st = pd.read_csv(exp / "m5_m6_per_alert.csv")
    rng = np.random.default_rng(0)
    rows = []
    for name, d in [("все", st), ("|E_v| > 10", st[st.n_edges > 10])]:
        x = d.jaccard_seeds.dropna().to_numpy()
        med = np.median(rng.choice(x, (2000, len(x))), axis=1)
        rows.append(dict(подвыборка=name, n=len(x), медиана=round(float(np.median(x)), 3),
                         ci=f"[{np.quantile(med, 0.025):.3f}; {np.quantile(med, 0.975):.3f}]",
                         доля_не_ниже_05=round(float((x >= 0.5).mean()), 3)))
    jac = pd.DataFrame(rows)
    jac.to_csv(exp / "m5_jaccard_ci.csv", index=False)
    print("\n== M-5: Жаккар top-5 между сидами, бутстрап-интервал медианы")
    print(jac.to_string(index=False))

    al = pd.read_csv(exp / "alerts.csv")
    tot = al.m2_local.abs() + al.m2_aggregated.abs() + al.m2_edges.abs()
    al = al.assign(sh_edges=al.m2_edges.abs() / tot, sh_local=al.m2_local.abs() / tot,
                   sh_agg=al.m2_aggregated.abs() / tot)
    by = al.groupby("step").agg(n=("v", "size"), граф=("sh_edges", "median"), локальные=("sh_local", "median"),
                                агрегаты=("sh_agg", "median")).round(3).reset_index()
    by.to_csv(exp / "m2_by_step.csv", index=False)
    print("\n== Б.1: медианные доли М2 по шагам (все алерты)")
    print(by.to_string(index=False))
    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин. Файлы: {exp}")


if __name__ == "__main__":
    main()
