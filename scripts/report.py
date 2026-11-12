#!/usr/bin/env python3
"""Build results/summary.json, results/table.md and docs/replan_curve.png from results/eval/*.json.

Each eval file is LeRobot's eval_info.json for one run: 4 LIBERO suites x 10 tasks x 10 episodes. Run names:
<model>_all_nas<k>[_s2000], where nas = n_action_steps (actions executed per prediction before re-planning)
and the eval seed is 1000 unless suffixed _s2000. Models: published (lerobot/smolvla_libero), ft (our
expert-only fine-tune), full (our full fine-tune).

The published model's first seed-1000 runs at 50 and 10 were split by suite and are combined (SPLIT).
Usage: python3 scripts/report.py   (needs numpy + matplotlib)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

R, DOCS = Path("results"), Path("docs")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_NAMES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
MODELS = {"full": "Ours: full fine-tune", "published": "LeRobot's published smolvla_libero",
          "ft": "Ours: expert-only (VLM frozen)"}
COLORS = {"full": "#2a78d6", "published": "#eb6834", "ft": "#1baf7a"}  # reference palette slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
PAPER = {"Spatial": 90, "Object": 96, "Goal": 92, "Long": 71, "avg": 87.3}  # arXiv 2506.01844, Table 2 (0.45B)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


# The published model's first seed-1000 runs were split by suite (Spatial, then Object/Goal/Long).
SPLIT = {"published_all_nas50": ["published_spatial", "published_ogl"],
         "published_all_nas10": ["published_spatial_nas10", "published_ogl_nas10"]}


def per_task(name):
    if name in SPLIT:
        return [t for part in SPLIT[name] for t in json.load(open(R / "eval" / f"{part}.json"))["per_task"]]
    return json.load(open(R / "eval" / f"{name}.json"))["per_task"]


runs = {}
names = sorted({f.stem for f in (R / "eval").glob("*_all_nas*.json")} | set(SPLIT))
for name in names:
    model, rest = name.split("_all_nas")
    model = {"expert": "ft"}.get(model, model)  # seed-2000 expert-only runs were named expert_*
    nas, seed = (rest.split("_s") + ["1000"])[:2]
    per = {}
    for t in per_task(name):
        s = per.setdefault(t["task_group"], [0, 0])
        s[0] += sum(t["metrics"]["successes"])
        s[1] += len(t["metrics"]["successes"])
    k, n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    runs[name] = {"model": model, "n_action_steps": int(nas), "seed": int(seed),
                  "suites": {SUITE_NAMES[g]: round(100 * per[g][0] / per[g][1], 1) for g in SUITES},
                  "avg": round(100 * k / n, 2), "episodes": n, "ci95": wilson(k, n)}

json.dump({"paper_smolvla_0.45B": PAPER, "runs": runs}, open(R / "summary.json", "w"), indent=2)

lines = ["| Model | Re-plan every (actions) | Eval seed | Spatial | Object | Goal | Long | **Average** (95% CI) |",
         "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    for r in sorted((v for v in runs.values() if v["model"] == m), key=lambda v: (v["seed"], -v["n_action_steps"])):
        s = r["suites"]
        lines.append(f"| {MODELS[m]} | {r['n_action_steps']} | {r['seed']} | {s['Spatial']:.0f} | {s['Object']:.0f} | "
                     f"{s['Goal']:.0f} | {s['Long']:.0f} | **{r['avg']:.1f}** ({r['ci95'][0]:.0f}–{r['ci95'][1]:.0f}) |")
lines.append(f"| SmolVLA paper (0.45B), re-plans every action | 1 | — | {PAPER['Spatial']} | {PAPER['Object']} | "
             f"{PAPER['Goal']} | {PAPER['Long']} | **{PAPER['avg']}** |")
(R / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# chart: average success vs re-planning interval, one line per model (seed 1000), seed-2000 checks as hollow marks
fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
for m, label in MODELS.items():
    pts = sorted((v["n_action_steps"], v["avg"]) for v in runs.values() if v["model"] == m and v["seed"] == 1000)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=COLORS[m], lw=2, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    chk = [(v["n_action_steps"], v["avg"]) for v in runs.values() if v["model"] == m and v["seed"] == 2000]
    ax.scatter([c[0] for c in chk], [c[1] for c in chk], s=70, facecolors="none", edgecolors=COLORS[m],
               linewidths=2, zorder=4)
ax.axhline(PAPER["avg"], color=INK2, lw=1, ls="--")
ax.text(49, PAPER["avg"] + 0.8, "SmolVLA paper, 0.45B (87.3%)", color=INK2, fontsize=9, ha="right")
ax.scatter([], [], s=70, facecolors="none", edgecolors=INK2, linewidths=2, label="re-check on a fresh seed (2000)")
ax.set_xscale("log")
ax.set_xticks([5, 10, 25, 50])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xticklabels(["5", "10", "25", "50\n(default)"])
ax.set_xlim(4.2, 60)
ax.set_ylim(55, 100)
ax.set_xlabel("Actions executed per prediction before re-planning (log scale)", color=INK2)
ax.set_ylabel("LIBERO success, average of 4 suites (%)", color=INK2)
ax.set_title("Re-planning every 10 actions instead of 50: +14 to +22 points for every model", loc="left", color=INK, fontsize=12, pad=10)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
fig.tight_layout()
DOCS.mkdir(exist_ok=True)
fig.savefig(DOCS / "replan_curve.png", dpi=150, facecolor=SURFACE)
print("wrote", DOCS / "replan_curve.png")
