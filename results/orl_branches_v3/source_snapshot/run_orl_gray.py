"""ORL Table II 前两行与 Fig. 7(a)；原文未公开细节均写入 protocol.json。"""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageOps
import scipy

from csa import CSA, PCA, LDA, nearest_neighbor


def load_orl(root):
    images, labels, names = [], [], []
    digest = hashlib.sha256()
    for person in range(1, 41):
        for number in range(1, 11):
            name = f"s{person}/{number}.pgm"
            path = root / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
            with Image.open(path) as image:
                if image.size != (92, 112):
                    raise ValueError(f"Unexpected raw ORL size: {name}")
                # Pillow resize 的参数是 (宽,高)，目标数组为 [56,46]。
                resized = image.convert("L").resize((46, 56), Image.Resampling.BILINEAR)
                images.append(np.asarray(ImageOps.equalize(resized), dtype=np.float64))
            labels.append(person)
            names.append(name)
    return np.stack(images), np.array(labels), names, digest.hexdigest()


def split_orl(labels, seed):
    rng = np.random.default_rng(seed)
    train, probe = [], []
    for person in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == person))
        train.extend(indices[:2])
        probe.extend(indices[2:])
    train, probe = np.array(train), np.array(probe)
    assert len(train) == 80 and len(probe) == 320
    assert not np.intersect1d(train, probe).size
    assert np.array_equal(np.sort(np.r_[train, probe]), np.arange(400))
    return train, probe


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_root, output, seeds):
    # 每次运行要求新目录，避免覆盖旧实验或混入旧结果。
    output.mkdir(parents=True, exist_ok=False)
    x, labels, names, raw_hash = load_orl(data_root)
    protocol = {
        "status": "partial reproduction with documented protocol differences",
        "scope": "ORL Table II gray rows / Figure 7(a) only; no Gabor experiment",
        "run_time_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "preprocess": "Pillow bilinear 112x92 -> 56x46, then ImageOps.equalize; float64 [0,255]",
        "split": "NumPy default_rng(seed); per subject first 2 permuted indices for training/gallery, other 8 probe",
        "seeds": seeds, "centering_variants": [False, True],
        "csa": {"ranks": [11, 2], "init": "identity truncated", "max_iter": 3, "epsilon": 0.001},
        "pca_components": 30,
        "lda": "Fisher generalized eigensystem via positive-Sw whitening; W.T Sw W=I; no ridge or column unit normalization",
        "classifier": "Euclidean 1-NN; no feature normalization; ties use first gallery index",
        "parameter_selection": "fixed paper configurations; dimension curves descriptive only, no probe-based parameter selection",
        "paper_targets_percent": {"PCA+LDA (Gray)": 82.2, "CSA-2+LDA": 84.1},
        "unknown_original_details": ["split indices", "resize interpolation", "histogram equalization implementation", "centering", "CSA initialization", "LDA solver/vector normalization", "NN metric"],
        "raw_data_sha256": raw_hash,
        "processed_data_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
    }
    for filename in ["csa.py", "run_orl_gray.py"]:
        protocol.setdefault("code_sha256", {})[filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, curves, predictions, histories, split_rows = [], [], [], [], []
    for seed in seeds:
        train, probe = split_orl(labels, seed)
        for partition, indices in [("train_gallery", train), ("probe", probe)]:
            split_rows.extend({"seed": seed, "partition": partition, "order": order,
                               "index": int(index), "file": names[index], "label": int(labels[index])}
                              for order, index in enumerate(indices))
        for center in [False, True]:
            models = [("PCA+LDA (Gray)", PCA(30, center=center), 18, 82.2),
                      ("CSA-2+LDA", CSA((11, 2), center=center), 16, 84.1)]
            for method, model, target_dim, paper in models:
                model.fit(x[train])
                gallery = model.transform(x[train]).reshape(80, -1)
                probes = model.transform(x[probe]).reshape(320, -1)
                lda = LDA().fit(gallery, labels[train])
                for dimension in range(1, lda.max_components + 1):
                    predicted, nearest, distance = nearest_neighbor(
                        lda.transform(gallery, dimension), labels[train], lda.transform(probes, dimension))
                    accuracy = float(100 * np.mean(predicted == labels[probe]))
                    curves.append({"seed": seed, "center": center, "method": method,
                                   "lda_dim": dimension, "accuracy_percent": accuracy})
                    if dimension == target_dim:
                        rows.append({"seed": seed, "center": center, "method": method,
                                     "pre_dims": "30" if isinstance(model, PCA) else "11x2",
                                     "lda_dim": dimension, "accuracy_percent": accuracy,
                                     "paper_percent": paper, "difference_pp": accuracy - paper,
                                     "correct": int(np.sum(predicted == labels[probe])), "probes": 320,
                                     "sw_rank": lda.sw_rank,
                                     "status": "protocol_differences"})
                        for index, prediction, neighbor, dist in zip(probe, predicted, nearest, distance):
                            predictions.append({"seed": seed, "center": center, "method": method,
                                                "probe": names[index], "true": int(labels[index]),
                                                "prediction": int(prediction), "nearest_gallery": names[train[neighbor]],
                                                "distance": float(dist), "correct": int(prediction == labels[index])})
                        print(f"seed={seed} center={center} {method}: {accuracy:.4f}% (paper {paper:.1f}%)", flush=True)
                if isinstance(model, CSA):
                    histories.append({"seed": seed, "center": center, "initial_sse": model.initial_sse,
                                      "history": model.history, "stop_reason": model.stop_reason})
    write_csv(output / "fixed_config_results.csv", rows)
    write_csv(output / "lda_dimension_curves.csv", curves)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "splits.csv", split_rows)
    (output / "csa_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    summary = []
    for center in [False, True]:
        for method in ["PCA+LDA (Gray)", "CSA-2+LDA"]:
            selected = [r for r in rows if r["center"] == center and r["method"] == method]
            values = [r["accuracy_percent"] for r in selected]
            summary.append({"center": center, "method": method, "runs": len(values),
                            "mean_percent": float(np.mean(values)),
                            "sample_std_pp": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "paper_percent": selected[0]["paper_percent"]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, center in zip(axes, [False, True]):
        for method, style in [("PCA+LDA (Gray)", "--"), ("CSA-2+LDA", "-")]:
            selected = [r for r in curves if r["center"] == center and r["method"] == method]
            dimensions = sorted({r["lda_dim"] for r in selected})
            values = np.array([[r["accuracy_percent"] for r in selected if r["lda_dim"] == d] for d in dimensions])
            mean = values.mean(axis=1)
            axis.plot(dimensions, mean, style, label=method)
            if len(seeds) > 1:
                std = values.std(axis=1, ddof=1)
                axis.fill_between(dimensions, mean - std, mean + std, alpha=0.15)
        axis.set(title=f"Training mean subtraction: {center}", xlabel="LDA dimension")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Top-one accuracy (%)")
    fig.suptitle("ORL gray: fixed pre-dimensions, documented protocol differences\nMean over fixed seeds; shading = sample standard deviation")
    fig.tight_layout()
    fig.savefig(output / "orl_gray_lda_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base / "data/raw/orl")
    parser.add_argument("--output", type=Path, default=base / "results/orl_gray_initial")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    run(args.data_root, args.output, args.seeds)
