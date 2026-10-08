"""Plot saved ORL scores without training or selecting any new configuration."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    source = base / "results/orl_gray_improved/test_results.csv"
    with source.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    levels = ["original_default", "plausible_fixed", "protocol_selected", "grid_selected"]
    captions = ["Previous\ndefaults", "Fixed\nhypothesis", "Protocol\nselected", "Protocol + dims\nselected"]
    colors = {"PCA+LDA": "#2878B5", "CSA-2+LDA": "#DF7C22"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    positions = np.arange(4)
    for method, offset in [("PCA+LDA", -0.18), ("CSA-2+LDA", 0.18)]:
        values = [[float(r["test_percent"]) for r in rows if r["method"] == method and r["level"] == level] for level in levels]
        assert all(len(v) == 5 for v in values)
        means = np.mean(values, axis=1)
        stds = np.std(values, axis=1, ddof=1)
        bars = axes[0].bar(positions + offset, means, width=0.36, yerr=stds,
                           capsize=3, color=colors[method], label=method)
        for bar, mean, std in zip(bars, means, stds):
            axes[0].text(bar.get_x() + bar.get_width()/2, mean + std + 1.2,
                         f"{mean:.2f}", ha="center", fontsize=9)
    axes[0].set(xticks=positions, xticklabels=captions, ylim=(0, 100),
                ylabel="Test accuracy (%)", title="Saved test means; error bars = split SD")
    axes[0].legend(loc="lower right")

    seeds = sorted({int(r["seed"]) for r in rows})
    for level, label, color in zip(levels, ["Previous defaults", "Fixed hypothesis", "Protocol selected", "Protocol + dims selected"],
                                   ["#777777", "#2878B5", "#6A50A7", "#DF7C22"]):
        values = [next(float(r["test_percent"]) for r in rows if r["method"] == "CSA-2+LDA" and r["level"] == level and int(r["seed"]) == seed) for seed in seeds]
        axes[1].plot(seeds, values, marker="o", color=color, label=label)
    # Paper values are reference points, not measurements on our held-out splits.
    axes[1].axhline(84.1, color="black", linestyle="--", linewidth=1.2,
                    label="Paper CSA 84.1% (different protocol)")
    axes[1].set(xticks=seeds, ylim=(60, 95), xlabel="Split seed", ylabel="Test accuracy (%)",
                title="CSA by split (axis: 60-95%)")
    axes[1].legend(loc="lower right", fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("ORL gray: 2 train / 2 validation / 6 test per subject; seeds 10-14\n"
                 "Paper Table II: PCA 82.2%, CSA 84.1%; 2 train / 8 probe, unknown original split", fontsize=12)
    fig.tight_layout()
    output = base / "results/visual_comparison"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "current_orl_comparison.png", dpi=180)
    plt.close(fig)
    print(output / "current_orl_comparison.png")


if __name__ == "__main__":
    main()
