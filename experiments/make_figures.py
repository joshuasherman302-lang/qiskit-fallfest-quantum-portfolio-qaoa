"""Turn results/*.json into figures/*.png. No numbers are computed here."""
from common import RESULTS, ROOT

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def finish(fig, name, legend_from=None, title=None):
    """Lay out with room reserved for a figure-level title (top) and legend (bottom)."""
    top, bottom = (0.9 if title else 1.0), (0.1 if legend_from is not None else 0.0)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", y=0.985)
    if legend_from is not None:
        handles, labels = legend_from.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(FIGURES / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def fig_ceiling():
    data = load("e02_ansatz_ceiling")
    series = [("x_penalty", "X mixer + penalty", ORANGE), ("xy_spread", "XY, basis-state start", AQUA),
              ("xy_pairs", "XY, paired start", BLUE), ("xy_dicke", "XY, Dicke start", YELLOW)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, n in zip(axes, (4, 6, 8)):
        cells = [c for c in data["cells"] if c["n"] == n]
        for key, label, color in series:
            pts = sorted((c for c in cells if c["variant"] == key), key=lambda c: c["p"])
            xs = range(len(pts))
            med = [c["ratio"]["median"] for c in pts]
            err = [[m - c["ratio"]["lo"] for m, c in zip(med, pts)],
                   [c["ratio"]["hi"] - m for m, c in zip(med, pts)]]
            ax.errorbar(xs, med, yerr=err, color=color, marker="o", capsize=3, label=label)
        ax.axhline(cells[0]["ratio_uniform"], color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(2.05, cells[0]["ratio_uniform"] - 0.035, "guessing", color=MUTED, ha="right", fontsize=9)
        ax.set_xticks(range(3), ["p=1", "p=2", "p=4"])
        ax.set_title(f"{n} assets")
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("cost score (1 = always optimal)")
    finish(fig, "ansatz_ceiling", legend_from=axes[0],
           title="With ideal tuning, the penalty circuit falls to guessing as assets are added")


def fig_tuners():
    data = load("e03_tuners")
    find = lambda tuner, start, evals: next(t["ratio"] for t in data["tuners"]
                                            if (t["tuner"], t["start"], t["evals"]) == (tuner, start, evals))
    ppo = data["ppo"]["context"]["ratio_median_over_seeds"]
    rows = [
        ("No tuning (start state)", data["neutral_start"]["ratio"], "0 evaluations"),
        ("Random search", find("random", "neutral", 200), "200 evaluations"),
        ("SPSA, Qiskit default step", find("spsa", "neutral", 200), "200 evaluations"),
        ("SPSA, step chosen on training data", find("spsa_tuned", "random_small", 200), "200 evaluations"),
        ("Bayesian optimisation", find("bayes", "neutral", 200), "200 evaluations"),
        ("PPO policy, zero-shot", {"median": ppo["mean"], "lo": ppo["min"], "hi": ppo["max"]}, "0 evaluations"),
        ("PPO start + SPSA (chosen step)", find("spsa_tuned", "ppo", 100), "100 evaluations"),
        ("PPO start + Bayesian", find("bayes", "ppo", 100), "100 evaluations"),
        ("Best found with exact tuning", data["ceiling_fourier4"]["ratio"], "reference"),
    ][::-1]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for y, (label, r, note) in enumerate(rows):
        ax.errorbar(r["median"], y, xerr=[[r["median"] - r["lo"]], [r["hi"] - r["median"]]],
                    color=BLUE, marker="o", capsize=3)
        ax.text(0.925, y, note, color=MUTED, fontsize=8.5, va="center")
    ax.axvline(data["uniform_guess_ratio"], color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(data["uniform_guess_ratio"] + 0.004, 0.45, "guessing", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("cost score, 20 held-out 2024-25 instances (median, 95% interval; PPO: mean and range of 5 seeds)")
    ax.set_title("Angle tuning on 8 assets, per-instance circuit evaluations at right")
    ax.grid(axis="y", visible=False)
    finish(fig, "tuners")


def fig_gates():
    rows = load("e04_hardware")["gate_costs"]
    ns = [r["n"] for r in rows]
    series = [("layer_naive_median", "one layer, transpiler routing", ORANGE),
              ("layer_swap_network", "one layer, swap network", BLUE),
              ("layer_fields_only", "one layer, fields only", AQUA),
              ("prep_dicke", "Dicke start", YELLOW), ("prep_pairs", "paired start", MAGENTA)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, key, title in ((axes[0], "two_qubit_gates", "Two-qubit gates"),
                           (axes[1], "two_qubit_depth", "Two-qubit depth")):
        for name, label, color in series:
            ys = [r[name][key] for r in rows]
            ax.plot(ns, ys, color=color, marker="o", label=label)
            # the two routing lines end close together: label one above, one below
            va = {"layer_naive_median": "bottom", "layer_swap_network": "top"}.get(name, "center")
            ax.text(ns[-1] + 0.3, ys[-1], str(ys[-1]), color=MUTED, fontsize=8.5, va=va)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xlim(3.5, 18)
        ax.set_xlabel("assets (qubits)")
        ax.set_title(f"{title} on ibm_brisbane's layout")
    axes[0].legend(fontsize=8.5, loc="upper left")
    finish(fig, "gate_costs")


def fig_noisy():
    data = load("e04_hardware")
    designs = []
    for c in data["noisy"]:
        if c["design"] not in designs:
            designs.append(c["design"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, n in zip(axes, (4, 6, 8)):
        cells = {c["design"]: c for c in data["noisy"] if c["n"] == n}
        for y, d in enumerate(designs[::-1]):
            c = cells[d]
            for key, color, marker, off in (("ideal_ratio", AQUA, "D", 0.14), ("mitigated_ratio", BLUE, "o", -0.14)):
                r = c[key]
                ax.errorbar(r["median"], y + off, xerr=[[r["median"] - r["lo"]], [r["hi"] - r["median"]]],
                            color=color, marker=marker, capsize=2.5,
                            label={"ideal_ratio": "noise-free", "mitigated_ratio": "Brisbane noise, mitigated"}[key] if y == 0 else None)
            ax.text(1.03, y, f"{c['two_qubit_gates']} gates", color=MUTED, fontsize=8, va="center")
        guess = list(cells.values())[0]["ratio_uniform"]
        ax.axvline(guess, color=MUTED, linestyle="--", linewidth=1.2)
        ax.set_title(f"{n} assets")
        ax.set_xlim(0.4, 1.16)
        ax.set_yticks(range(len(designs)), designs[::-1])
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("cost score after budget post-selection (dashed line = guessing)")
    finish(fig, "noisy_brisbane", legend_from=axes[0],
           title="Under ibm_brisbane's noise model only the shallow designs keep their advantage")


def fig_dial():
    rows = load("e05_classical")["hardness_dial"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for n, color in ((8, BLUE), (12, ORANGE)):
        pts = [r for r in rows if r["n"] == n]
        ys = [r["sorting_finds_optimum"] for r in pts]
        ax.plot(range(len(pts)), ys, color=color, marker="o", label=f"{n} assets")
        ax.text(len(pts) - 0.9, ys[-1], f"{ys[-1]:.0%}", color=MUTED, fontsize=9, va="center")
    ax.set_xticks(range(5), ["1", "3", "10", "30", "100"])
    ax.set_xlim(-0.2, 4.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("risk aversion")
    ax.set_ylabel("share of instances solved by sorting alone")
    ax.set_title("Sorting solves the easy regime; interactions matter as risk dominates")
    ax.legend()
    finish(fig, "hardness_dial")


if __name__ == "__main__":
    for make in (fig_ceiling, fig_tuners, fig_gates, fig_noisy, fig_dial):
        try:
            make()
        except FileNotFoundError as missing:
            print(f"skipped {make.__name__}: {missing}")
