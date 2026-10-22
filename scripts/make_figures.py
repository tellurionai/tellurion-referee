"""Figure 1 and the results fact sheet, generated from a results file (T42, T43, T49).

    python scripts/make_figures.py --input results/test_metrics.json

Writes figures/figure1.png + figures/figure1.svg and docs/results_fact_sheet.md.
Every number comes from the input file; nothing is typed by hand. Run it again after
any new evaluation and commit the outputs together with the results they came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "referee": "Referee (ours)",
    "b1_ensemble_uncertainty": "AI's own confidence (B1)",
    "b2_residual": "Prediction gap (B2)",
    "b3_majority_verdict": "Most common answer (B3)",
    "oracle": "Oracle (sees the truth)",
}
VERDICT_NAMES = {"trust": "Trust", "model_error": "AI is wrong",
                 "instrument_fault": "Sensor is wrong", "discovery": "Something new"}

# Reference palette (dataviz skill): one accent for ours, recessive ink for the rest.
ACCENT, MUTED = "#2a78d6", "#8a8984"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci_pct(c) -> str:
    lo, hi = c["ci95"]
    return "n/a" if lo is None else f"{100 * lo:.1f}–{100 * hi:.1f}%"


def num(x, d=3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def ci_num(c, d=3) -> str:
    lo, hi = c["ci95"]
    return "n/a" if lo is None else f"{lo:.{d}f}–{hi:.{d}f}"


# --- Figure 1 ----------------------------------------------------------------------

def figure(r: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11.5, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55, left=0.2, right=0.97, top=0.78, bottom=0.14)
    m, n = r["methods"], r["summary"]["n_trajectories"]

    # A: missed problems, same false-alarm budget
    ax = fig.add_subplot(gs[0])
    rows = ["referee", "b2_residual", "b1_ensemble_uncertainty"]
    for y, key in enumerate(rows):
        d = m[key]["detection"]
        lo, hi = d["ci"]["miss_rate"]["ci95"]
        color = ACCENT if key == "referee" else MUTED
        ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([100 * d["miss_rate"]], [y], s=90, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(100 * hi + 1.2, y, f"{100 * d['miss_rate']:.1f}% missed", va="center", fontsize=9.5,
                color=INK if key == "referee" else INK_2, fontweight="bold" if key == "referee" else "normal")
    ax.set_yticks(range(len(rows)), [f"{NAMES[k]}\nfalse alarms {pct(m[k]['detection']['false_alarm_rate'])}"
                                     for k in rows])
    for lab, key in zip(ax.get_yticklabels(), rows):
        lab.set_color(INK if key == "referee" else INK_2)
        lab.set_fontweight("bold" if key == "referee" else "normal")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 45)
    ax.set_xlabel("Real problems missed (%), 95% CI", fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A  Noticing problems", loc="left", fontsize=11, fontweight="bold", pad=10)

    # B: referee diagnosis, row-normalised
    bx = fig.add_subplot(gs[1])
    g = m["referee"]["diagnosis"]
    cm = np.array(g["confusion"], dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    bx.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            dark = share[i, j] > 0.55
            bx.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9.5,
                    color=SURFACE if dark else INK, fontweight="bold" if i == j else "normal")
    labels = [VERDICT_NAMES[v] for v in g["labels"]]
    bx.set_xticks(range(4), labels, rotation=30, ha="right", fontsize=9)
    bx.set_yticks(range(4), [f"{lab}  ({pct(g['recall'][v])})" for lab, v in zip(labels, g["labels"])], fontsize=9)
    bx.set_xlabel("Referee's verdict", fontsize=9.5)
    bx.set_ylabel("True cause  (share named correctly)", fontsize=9.5)
    bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    bx.set_xticks(np.arange(-0.5, 4), minor=True)
    bx.set_yticks(np.arange(-0.5, 4), minor=True)
    bx.grid(which="minor", color=SURFACE, lw=2)
    bx.tick_params(which="minor", length=0)
    bx.set_title("B  Naming the cause (referee)", loc="left", fontsize=11, fontweight="bold", pad=10)

    b1 = m["b1_ensemble_uncertainty"]["detection"]["miss_rate"]
    ref = m["referee"]["detection"]["miss_rate"]
    fig.suptitle(f"At a similar false-alarm rate, the AI's own confidence missed {100 * b1:.0f}% of real problems; "
                 f"the referee missed {100 * ref:.1f}%", x=0.02, y=0.97, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.02, 0.885, f"Locked test set, {n} simulated pendulum runs never seen during development "
             f"({r['summary']['n_clean']} normal). Thresholds tuned on a separate calibration set; "
             f"95% CIs resample whole runs.", ha="left", fontsize=9, color=INK_2)
    fig.text(0.02, 0.025, f"Source: results/{r['split']}_metrics.json · commit {str(r['git']['commit'])[:7]} · "
             f"model {r['model']['checkpoint_sha256'][0][:8]}", ha="left", fontsize=8, color=INK_2)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure1.png", dpi=200)
    fig.savefig(out / "figure1.svg")
    plt.close(fig)


# --- fact sheet -----------------------------------------------------------------------

def fact_sheet(r: dict, src: str) -> str:
    m, s = r["methods"], r["summary"]
    ref, b1 = m["referee"], m["b1_ensemble_uncertainty"]
    rd, bd = ref["detection"], b1["detection"]
    g = ref["diagnosis"]
    lines = [
        "# Results fact sheet",
        "",
        f"Generated by `scripts/make_figures.py` from `{src}`. **Do not edit by hand**: every number in the",
        "one-pager, the video and the demo must match this page. Regenerate after any new evaluation.",
        "",
        "## Provenance",
        "",
        f"- Split: **{r['split']}** ({s['n_trajectories']} runs, {s['n_clean']} normal). "
        f"Answer key counts: " + ", ".join(f"{VERDICT_NAMES[k]} {v}" for k, v in s["verdict_counts"].items()) + ".",
        f"- Code: commit `{str(r['git']['commit'])[:7]}` (clean tree: {not r['git']['dirty']})"
        + (" · tag `v0.1-hackathon-test`" if r["split"] == "test" else ""),
        f"- Model: `{r['model']['dir']}`, member hashes {', '.join(h[:8] for h in r['model']['checkpoint_sha256'])}",
        f"- Data: `{r['dataset']['file']}` (SHA256 `{r['dataset']['sha256'][:16]}…`), seed {r['seed']}",
        f"- Run at {r['created_at']}. Thresholds fit on {r['settings']['thresholds_fit_on']}; "
        f"target false-alarm rate {pct(r['settings']['target_false_alarm_rate'])}; a detection = "
        f"{r['settings']['sustain_steps']} consecutive flagged steps; CIs: {r['settings']['bootstrap_samples']} "
        "bootstrap resamples of whole runs.",
        "",
        "## Headline sentences (copy exactly)",
        "",
        f"- At a similar false-alarm rate ({pct(rd['false_alarm_rate'])} vs {pct(bd['false_alarm_rate'])} of normal runs), "
        f"the AI's own confidence missed **{pct(bd['miss_rate'])}** of real problems (95% CI {ci_pct(bd['ci']['miss_rate'])}); "
        f"the referee missed **{pct(rd['miss_rate'])}** ({ci_pct(rd['ci']['miss_rate'])}).",
        f"- Detection AUROC: referee **{num(rd['auroc'])}** ({ci_num(rd['ci']['auroc'])}) vs "
        f"AI's own confidence **{num(bd['auroc'])}** ({ci_num(bd['ci']['auroc'])}).",
        f"- Naming the cause (macro-F1, 4 verdicts): referee **{num(g['macro_f1'])}** ({ci_num(g['macro_f1_ci'])}) vs "
        f"always guessing the most common answer **{num(m['b3_majority_verdict']['diagnosis']['macro_f1'])}**.",
        f"- The referee names the cause correctly for " + ", ".join(
            f"{VERDICT_NAMES[v]} {pct(g['recall'][v])}" for v in g["labels"]) + ".",
        "",
        "## Detection (noticing something is wrong)",
        "",
        "| Method | AUROC (95% CI) | Missed (95% CI) | False alarms, normal runs (95% CI) | Median delay (steps) | Caught when confidently wrong |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("referee", "b1_ensemble_uncertainty", "b2_residual", "oracle"):
        d = m[k]["detection"]
        lines.append(f"| {NAMES[k]} | {num(d['auroc'])} ({ci_num(d['ci']['auroc'])}) | {pct(d['miss_rate'])} "
                     f"({ci_pct(d['ci']['miss_rate'])}) | {pct(d['false_alarm_rate'])} ({ci_pct(d['ci']['false_alarm_rate'])}) | "
                     f"{d['delay_steps_median']} | {pct(m[k]['confidently_wrong_recall'])} |")
    lines += [
        "",
        f"N: {rd['n_positive']} runs with a real problem, {rd['n_clean']} normal runs. One step = "
        f"{r['dataset']['metadata']['dt']} s. Detected share by fault severity (referee): "
        + ", ".join(f"{k} {pct(v)}" for k, v in rd["detected_rate_by_severity"].items() if k != "none") + ".",
        "",
        "## Diagnosis (naming the cause)",
        "",
        "| Method | Macro-F1 (95% CI) | Accuracy |",
        "|---|---|---|",
    ]
    for k in ("referee", "b2_residual", "b1_ensemble_uncertainty", "b3_majority_verdict"):
        dg = m[k]["diagnosis"]
        lines.append(f"| {NAMES[k]} | {num(dg['macro_f1'])} ({ci_num(dg['macro_f1_ci'])}) | {pct(dg['accuracy'])} |")
    labels = [VERDICT_NAMES[v] for v in g["labels"]]
    lines += ["", "Referee: true cause (rows) vs its verdict (columns), number of runs:", "",
              "| True cause ↓ / verdict → | " + " | ".join(labels) + " | Named correctly |",
              "|---|" + "---|" * (len(labels) + 1)]
    for v, lab, row in zip(g["labels"], labels, g["confusion"]):
        lines.append(f"| {lab} | " + " | ".join(str(x) for x in row) + f" | {pct(g['recall'][v])} |")
    by = g["by_perturbation_subtype"]
    lines += ["", "Referee accuracy by fault subtype: " + ", ".join(
        f"{k} {pct(v['accuracy'])} (n={v['n']})" for k, v in sorted(by.items())) + "."]
    lines += [
        "",
        "## Read these numbers correctly",
        "",
        f"- **Detection is the headline.** The referee's detection score is the prediction gap, so it matches the "
        f"simple gap alarm (B2, AUROC {num(m['b2_residual']['detection']['auroc'])}). Its added value is naming the cause.",
        f"- **Naming the cause is weak for sensor faults and new forces.** Main reason found: the referee names the cause "
        "at its first alarm, when its window is still mostly from before the fault. Next step: let it judge a few steps later.",
        "- **'Caught when confidently wrong'** is defined by low ensemble spread, so the AI's own confidence scores 0% "
        "on it by construction. Use the missed-problems comparison as the fair headline.",
        "- False-alarm rates were tuned to ≤ 5% on the calibration set; on the locked test they land slightly differently "
        "(reported as measured).",
        "- Everything is simulated (a single pendulum) so that every run has a known answer. Real systems are next.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="results/test_metrics.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--fact-sheet", default="docs/results_fact_sheet.md")
    a = ap.parse_args(argv)
    r = json.loads(Path(a.input).read_text(encoding="utf-8"))
    figure(r, Path(a.figures))
    Path(a.fact_sheet).write_text(fact_sheet(r, a.input), encoding="utf-8")
    print(f"wrote {a.figures}/figure1.png, {a.figures}/figure1.svg and {a.fact_sheet} from {a.input}")


if __name__ == "__main__":
    main()
