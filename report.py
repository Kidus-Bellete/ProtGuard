"""Aggregate results.json into a figure + markdown tables."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

R = json.load(open(os.path.join(M.OUT, "results.json")))
ARMS = ["A", "B", "C", "D", "E", "C0", "D0", "E0"]
LABEL = {"A": "A undefended", "B": "B +ProtScreener", "C": "C wrapper L1-L3", "D": "D C+steering", "E": "E C+LoRA reroute",
         "C0": "C0 L1+L2 only", "D0": "D0 L1+L2+steer (no probe)", "E0": "E0 L1+L2+LoRA (no probe)"}
COL = {"A": "#888888", "B": "#c98b2b", "C": "#2b7bb9", "D": "#4aa86b", "E": "#a04bb0",
       "C0": "#2b7bb9", "D0": "#4aa86b", "E0": "#a04bb0"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def agg(key):
    """arm -> dict(rounds x (mean, sd, pooled k, pooled n), runs, allowed, expansion)."""
    out = {}
    for a in ARMS:
        runs = R[key][a]; nq = None
        ev = np.array([[r["evasion"] for r in run] for run in runs])           # runs x rounds
        al = np.array([[r["allowed"] for r in run] for run in runs])
        ex = np.array([[r["expansion"] for r in run] for run in runs])
        succ = np.array([[r["succ"] for r in run] for run in runs])
        nq = int(round(succ[0, 0] / ev[0, 0])) if ev[0, 0] > 0 else None
        out[a] = dict(ev=ev, al=al, ex=ex, succ=succ, runs=len(runs))
    return out


def pooled_ci(ev_row_sum, n_total):
    return wilson(ev_row_sum, n_total)


per_round = None
for a in ARMS:           # infer queries/round from stored counts
    for run in R["red_S_adaptive"][a]:
        for r in run:
            if r["succ"] > 0 and r["evasion"] > 0:
                per_round = int(round(r["succ"] / r["evasion"])); break
        if per_round: break
    if per_round: break
per_round = per_round or 500

lines = []
lines.append("## Benign behaviour (400 benign requests; perplexity on 300 held-out natural benign sequences)\n")
lines.append("| Arm | FPR (benign blocked) | Benign generations passing checks | Perplexity | PPL vs undefended |")
lines.append("|---|---|---|---|---|")
bp = R["benign"]["A"]["ppl"]
for a in ARMS:
    b = R["benign"][a]
    lines.append(f"| {LABEL[a]} | {100*b['fpr']:.1f}% | {100*b['quality_pass']:.1f}% | {b['ppl']:.1f} | {b['ppl']/bp:.3f}x |")

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
for kidx, (key, title) in enumerate([("red_S_adaptive", "RQ1: adaptive attacker vs screened family S"),
                                     ("red_S_static", "Static (non-adaptive) attacker vs S")]):
    if key not in R:
        continue
    G = agg(key)
    lines.append(f"\n## {title} ({per_round} queries/round, 3 rounds)\n")
    lines.append("| Arm | runs | Evasion R1 | Evasion R2 | Evasion R3 | R3-R1 | Allowed rate R3 | Token expansion (mean) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        g = G[a]; m = g["ev"].mean(0); s = g["ev"].std(0); n = g["runs"]
        cells = []
        for r in range(3):
            lo, hi = wilson(g["succ"][:, r].sum(), per_round * n)
            cells.append(f"{100*m[r]:.1f}% ± {100*s[r]:.1f} [{100*lo:.1f},{100*hi:.1f}]")
        lines.append(f"| {LABEL[a]} | {n} | " + " | ".join(cells) + f" | {100*(m[2]-m[0]):+.1f} pts | {100*g['al'].mean(0)[2]:.0f}% | {g['ex'].mean():.2f}x |")
    if kidx == 0:
        a_ax = ax[0, 0]
        for a in ARMS:
            m = G[a]["ev"].mean(0) * 100; s = G[a]["ev"].std(0) * 100
            a_ax.errorbar([1, 2, 3], m, yerr=s, marker="o", color=COL[a], label=LABEL[a], capsize=3,
                          ls="--" if a.endswith("0") else "-", alpha=0.75 if a.endswith("0") else 1)
        a_ax.set_xticks([1, 2, 3]); a_ax.set_xlabel("red-team round"); a_ax.set_ylabel("evasion rate (% of 500 queries)")
        a_ax.set_title("RQ1: evasion of screened family S (adaptive attacker)"); a_ax.legend(fontsize=7)
        a_ax.grid(alpha=.3)

if "red_H_adaptive" in R:
    G = agg("red_H_adaptive")
    lines.append(f"\n## RQ2: adaptive attacker vs HELD-OUT family H (defenses trained on S only)\n")
    lines.append("| Arm | runs | Evasion R1 | Evasion R2 | Evasion R3 |")
    lines.append("|---|---|---|---|---|")
    for a in ARMS:
        g = G[a]; m = g["ev"].mean(0); s = g["ev"].std(0)
        lines.append(f"| {LABEL[a]} | {g['runs']} | " + " | ".join(f"{100*m[r]:.1f}% ± {100*s[r]:.1f}" for r in range(3)) + " |")
    a_ax = ax[0, 1]
    GS = agg("red_S_adaptive")
    x = np.arange(len(ARMS)); w = 0.38
    a_ax.bar(x - w/2, [GS[a]["ev"][:, 2].mean() * 100 for a in ARMS], w, label="S (trained on)", color="#2b7bb9")
    a_ax.bar(x + w/2, [G[a]["ev"][:, 2].mean() * 100 for a in ARMS], w, label="H (held out)", color="#e07b39")
    a_ax.set_xticks(x); a_ax.set_xticklabels(ARMS, fontsize=8); a_ax.set_ylabel("round-3 evasion (%)")
    a_ax.set_title("RQ2: generalisation to held-out family"); a_ax.legend(fontsize=8); a_ax.grid(alpha=.3, axis="y")

lines.append("\n## RQ2: probe detection of NATURAL sequences at the frozen threshold\n")
lines.append("| Arm | S_test (in-family) | H_test (held-out family) | Benign (false alarm) |")
lines.append("|---|---|---|---|")
for a in ["C", "D", "E"]:
    d = R["detect"][a]
    lines.append(f"| {LABEL[a]} | {100*d['S_test']:.0f}% | {100*d['H_test']:.0f}% | {100*d['benign']:.1f}% |")

# utility panels
a_ax = ax[1, 0]
x = np.arange(len(ARMS))
X = json.load(open(os.path.join(M.OUT, "extras.json")))
a_ax.bar(x - 0.2, [100 * R["benign"][a]["fpr"] for a in ARMS], 0.4, label="benign requests blocked (%)", color="#c0392b")
base_f = X["benign_fluency"]["A"]["ppl"]
a_ax.bar(x + 0.2, [100 * (X["benign_fluency"][a]["ppl"] / base_f - 1) for a in ARMS], 0.4, label="benign output perplexity vs A (% change)", color="#27ae60")
a_ax.set_xticks(x); a_ax.set_xticklabels(ARMS, fontsize=8)
a_ax.set_title("Benign cost (real Swiss-Prot)"); a_ax.legend(fontsize=7); a_ax.grid(alpha=.3, axis="y")
a_ax = ax[1, 1]
if "red_S_adaptive" in R and "red_S_static" in R:
    GA, GT = agg("red_S_adaptive"), agg("red_S_static")
    a_ax.bar(x - 0.2, [GT[a]["ev"].mean() * 100 for a in ARMS], 0.4, label="static attacker", color="#95a5a6")
    a_ax.bar(x + 0.2, [GA[a]["ev"].mean() * 100 for a in ARMS], 0.4, label="adaptive attacker", color="#2c3e50")
    a_ax.set_xticks(x); a_ax.set_xticklabels(ARMS, fontsize=8); a_ax.set_ylabel("mean evasion over 3 rounds (%)")
    a_ax.set_title("Does adaptation help the attacker?"); a_ax.legend(fontsize=8); a_ax.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(M.OUT, "results.png"), dpi=140)

# which families succeeded, and why queries were blocked (adaptive, S)
lines.append("\n## Where the adaptive attacker succeeded (round-3 hits per family, summed over runs, arm A and B)\n")
for a in ARMS:
    tot = {}
    for run in R["red_S_adaptive"][a]:
        for f, v in run[2]["fam_succ"].items():
            tot[f] = tot.get(f, 0) + v
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:5]
    lines.append(f"- {LABEL[a]}: " + ", ".join(f"{f}={v}" for f, v in top))
lines.append("\n## Why queries were blocked (round 3, adaptive vs S, summed over runs)\n")
for a in ARMS:
    rs = {}
    for run in R["red_S_adaptive"][a]:
        for k, v in run[2]["reasons"].items():
            rs[k or "allowed"] = rs.get(k or "allowed", 0) + v
    lines.append(f"- {LABEL[a]}: " + ", ".join(f"{k}={v}" for k, v in sorted(rs.items(), key=lambda kv: -kv[1])))
info = R["info"]
lines.append(f"\nSteering strength chosen (fraction of median activation norm): {info['steer_c']}; base benign perplexity (calib) {info['base_ppl']:.1f}")
lines.append(f"Oracle: {R['oracle']}")
open(os.path.join(M.OUT, "results_tables.md"), "w").write("\n".join(lines))
print("\n".join(lines))
