"""ORL 灰度改进：固定 2 train / 2 validation / 6 test，只用验证集选择。

保留原实现作为 control；未知协议与维数优化分别选择，不使用 test 选参。
本实验与论文 2 train / 8 probe 不同，不能充当原文数值复现。
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.distance import cdist

from csa import CSA, PCA, LDA
from run_orl_gray import write_csv


SEEDS = [10, 11, 12, 13, 14]
PROFILES = [("bilinear", "pillow"), ("bilinear", "cdf_min256"),
            ("lanczos", "pillow"), ("lanczos", "cdf_min256")]
PCA_DIMS = [30, 20, 40]
CSA_RANKS = [(11, 2), (8, 3), (6, 4), (5, 6), (6, 6), (10, 4)]
DISTANCES = [("unit_columns", "cosine", "train_mean"),
             ("unit_columns", "cosine", "zero"),
             ("unit_columns", "euclidean", "train_mean"),
             ("sw_identity", "cosine", "train_mean"),
             ("sw_identity", "cosine", "zero"),
             ("sw_identity", "euclidean", "train_mean")]


def equalize_cdf(image):
    """256级经验CDF并减去第一个非零CDF；不是声称 MATLAB histeq 的复制。"""
    array = np.asarray(image, dtype=np.uint8)
    cumulative = np.cumsum(np.bincount(array.ravel(), minlength=256))
    first = cumulative[np.flatnonzero(cumulative)[0]]
    if cumulative[-1] == first:
        return np.zeros(array.shape, dtype=np.float64)
    lookup = np.clip((cumulative - first) * 255.0 / (cumulative[-1] - first), 0, 255)
    return lookup[array]


def load_profiles(root):
    collections = {f"{a}_{b}": [] for a, b in PROFILES}
    names, labels = [], []
    digest = hashlib.sha256()
    for person in range(1, 41):
        for number in range(1, 11):
            name = f"s{person}/{number}.pgm"
            path = root / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
            with Image.open(path) as source:
                if source.size != (92, 112):
                    raise ValueError("Unexpected ORL raw size")
                for interpolation, equalization in PROFILES:
                    resampling = Image.Resampling.BILINEAR if interpolation == "bilinear" else Image.Resampling.LANCZOS
                    image = source.convert("L").resize((46, 56), resampling)
                    values = np.asarray(ImageOps.equalize(image), dtype=np.float64) if equalization == "pillow" else equalize_cdf(image)
                    collections[f"{interpolation}_{equalization}"].append(values)
            names.append(name)
            labels.append(person)
    return {key: np.stack(value) for key, value in collections.items()}, np.asarray(labels), names, digest.hexdigest()


def split(labels, seed):
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        partitions[0].extend(indices[:2])
        partitions[1].extend(indices[2:4])
        partitions[2].extend(indices[4:])
    train, validation, test = map(np.asarray, partitions)
    assert (len(train), len(validation), len(test)) == (80, 80, 240)
    assert len(set(train) | set(validation) | set(test)) == 400
    for partition, count in zip(partitions, [2, 2, 6]):
        assert all(np.sum(labels[partition] == c) == count for c in np.unique(labels))
    return train, validation, test


def model_options(method):
    if method == "PCA+LDA":
        return [(dimension, "svd", 0) for dimension in PCA_DIMS]
    return [(ranks, init, iterations) for ranks in CSA_RANKS
            for init in ["identity", "random"] for iterations in [3, 10]]


def classify(gallery, queries, labels, metric):
    distances = cdist(queries, gallery, metric)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite classification distance")
    nearest = np.argmin(distances, axis=1)
    return labels[nearest], nearest


def coordinates(features, lda, config):
    basis = lda.basis[:, :config["lda_dim"]]
    if config["scaling"] == "unit_columns":
        basis = basis / np.linalg.norm(basis, axis=0)
    origin = lda.mean if config["origin"] == "train_mean" else 0
    return (features - origin) @ basis


def candidate(config, model, lda, gallery_features, validation_correct):
    return {"config": config.copy(), "model": model, "lda": lda,
            "gallery_features": gallery_features, "validation_correct": validation_correct}


def evaluate(selected, data, train, test, labels, names):
    config, model, lda = selected["config"], selected["model"], selected["lda"]
    gallery = coordinates(selected["gallery_features"], lda, config)
    features = model.transform(data[test]).reshape(len(test), -1)
    queries = coordinates(features, lda, config)
    predicted, nearest = classify(gallery, queries, labels[train], config["metric"])
    details = [{"test_file": names[index], "true": int(labels[index]),
                "predicted": int(label), "nearest_gallery": names[train[neighbor]],
                "correct": int(label == labels[index])}
               for index, label, neighbor in zip(test, predicted, nearest)]
    return int(np.sum(predicted == labels[test])), details


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parent
    started = time.perf_counter()
    profiles, labels, names, raw_hash = load_profiles(base / "data/raw/orl")
    protocol = {
        "time_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "ORL gray optimization, not full/exact paper reproduction",
        "split_per_class": {"train_gallery": 2, "validation": 2, "test": 6},
        "seeds": SEEDS, "training_images": 80, "validation_images": 80, "test_images": 240,
        "profiles": PROFILES, "centering": [True, False], "pca_pre_dims": PCA_DIMS,
        "csa_pre_dims": CSA_RANKS, "csa_init": ["identity", "random"],
        "random_initialization_seed": 2026, "csa_max_iter": [3, 10], "epsilon": 0.001,
        "lda_dim_grid": "paper anchor first, then 8,12,16,18,24,30,39 and available maximum; invalid dims removed",
        "distance_configurations": DISTANCES,
        "selection": "highest validation correct count; ties retain first in declared iteration order",
        "test_used_for_selection": False,
        "comparators": {
            "original_default": "bilinear_pillow, center=False, paper dimensions, identity/3 CSA, Sw-normalized LDA + Euclidean, train-mean origin",
            "plausible_fixed": "bilinear_pillow, center=True, paper dimensions, identity/3 CSA, unit-column LDA + cosine, train-mean origin; hypothetical, not confirmed original",
            "protocol_selected": "validation selects unknown preprocessing/centering/init/scaling/metric/origin; paper pre/LDA dims and Tmax=3 fixed",
            "grid_selected": "validation selects protocol and bounded pre/LDA dimensions plus Tmax",
        },
        "limitations": "Same 400 images have been explored previously. Test is held out within each new split, not a new untouched dataset; repeated splits overlap. Test results do not influence this frozen grid.",
        "raw_data_sha256": raw_hash,
        "profile_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in profiles.items()},
        "code_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                        for name in ["csa.py", "improve_orl_gray.py"]},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    score_rows, predictions, split_rows, selections = [], [], [], []
    validation_count = 0
    with (output / "validation_search.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv
        fields = ["seed", "method", "profile", "center", "pre_dims", "init", "max_iter", "lda_dim", "scaling", "metric", "origin", "validation_correct", "validation_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in SEEDS:
            train, validation, test = split(labels, seed)
            for partition, indices in [("train_gallery", train), ("validation", validation), ("test", test)]:
                split_rows.extend({"seed": seed, "partition": partition, "order": order,
                                   "file": names[index], "index": int(index), "label": int(labels[index])}
                                  for order, index in enumerate(indices))
            for method in ["PCA+LDA", "CSA-2+LDA"]:
                chosen = {}
                anchor_pre, anchor_lda = (30, 18) if method == "PCA+LDA" else ((11, 2), 16)
                for profile, data in profiles.items():
                    for center in [True, False]:
                        for pre, init, iterations in model_options(method):
                            model = PCA(pre, center=center) if method == "PCA+LDA" else CSA(pre, center=center, init=init, seed=2026, max_iter=iterations)
                            model.fit(data[train])
                            gallery_features = model.transform(data[train]).reshape(80, -1)
                            query_features = model.transform(data[validation]).reshape(80, -1)
                            lda = LDA().fit(gallery_features, labels[train])
                            dimensions = list(dict.fromkeys(d for d in [anchor_lda, 8, 12, 16, 18, 24, 30, 39, lda.max_components] if 1 <= d <= lda.max_components))
                            for dimension in dimensions:
                                for scaling, metric, origin in DISTANCES:
                                    config = {"method": method, "profile": profile, "center": center,
                                              "pre_dims": pre, "init": init, "max_iter": iterations,
                                              "lda_dim": dimension, "scaling": scaling, "metric": metric, "origin": origin}
                                    predicted, _ = classify(coordinates(gallery_features, lda, config),
                                                            coordinates(query_features, lda, config), labels[train], metric)
                                    correct = int(np.sum(predicted == labels[validation]))
                                    row = {"seed": seed, **config, "validation_correct": correct, "validation_percent": 100 * correct / 80}
                                    row["pre_dims"] = str(pre)
                                    writer.writerow(row)
                                    validation_count += 1
                                    anchor = pre == anchor_pre and dimension == anchor_lda and (method == "PCA+LDA" or iterations == 3)
                                    levels = ["grid_selected"] + (["protocol_selected"] if anchor else [])
                                    fixed_model = anchor and profile == "bilinear_pillow" and (method == "PCA+LDA" or init == "identity")
                                    if fixed_model and not center and (scaling, metric, origin) == ("sw_identity", "euclidean", "train_mean"):
                                        chosen["original_default"] = candidate(config, model, lda, gallery_features, correct)
                                    if fixed_model and center and (scaling, metric, origin) == ("unit_columns", "cosine", "train_mean"):
                                        chosen["plausible_fixed"] = candidate(config, model, lda, gallery_features, correct)
                                    for level in levels:
                                        if level not in chosen or correct > chosen[level]["validation_correct"]:
                                            chosen[level] = candidate(config, model, lda, gallery_features, correct)
                assert len(chosen) == 4
                # 所有配置选择完成之后，才访问此方法的 test 特征/标签评分。
                for level in ["original_default", "plausible_fixed", "protocol_selected", "grid_selected"]:
                    selected = chosen[level]
                    correct, details = evaluate(selected, profiles[selected["config"]["profile"]], train, test, labels, names)
                    row = {"seed": seed, "method": method, "level": level,
                           "validation_correct": selected["validation_correct"],
                           "validation_percent": 100 * selected["validation_correct"] / 80,
                           "test_correct": correct, "test_count": 240, "test_percent": 100 * correct / 240}
                    score_rows.append(row)
                    predictions.extend({"seed": seed, "method": method, "level": level, **d} for d in details)
                    selections.append({"seed": seed, "method": method, "level": level, "config": selected["config"],
                                       "validation_percent": row["validation_percent"], "test_percent": row["test_percent"]})
                    print(f"seed={seed} {method} {level}: val={row['validation_percent']:.2f}% test={row['test_percent']:.2f}%", flush=True)
    write_csv(output / "test_results.csv", score_rows)
    write_csv(output / "test_predictions.csv", predictions)
    write_csv(output / "splits.csv", split_rows)
    (output / "selected_configs.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    summary = []
    for method in ["PCA+LDA", "CSA-2+LDA"]:
        for level in ["original_default", "plausible_fixed", "protocol_selected", "grid_selected"]:
            rows = [r for r in score_rows if r["method"] == method and r["level"] == level]
            values = [r["test_percent"] for r in rows]
            summary.append({"method": method, "level": level, "runs": len(rows),
                            "mean_test_percent": float(np.mean(values)), "sample_std_pp": float(np.std(values, ddof=1)),
                            "min_test_percent": min(values), "max_test_percent": max(values),
                            "mean_validation_percent": float(np.mean([r["validation_percent"] for r in rows]))})
    meta = {"validation_evaluations": validation_count, "wall_seconds": time.perf_counter() - started,
            "summary": summary}
    (output / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    levels = ["original_default", "plausible_fixed", "protocol_selected", "grid_selected"]
    positions = np.arange(4)
    for shift, method in [(-0.18, "PCA+LDA"), (0.18, "CSA-2+LDA")]:
        selected = [next(s for s in summary if s["method"] == method and s["level"] == level) for level in levels]
        axis.bar(positions + shift, [s["mean_test_percent"] for s in selected], 0.36,
                 yerr=[s["sample_std_pp"] for s in selected], capsize=4, label=method)
    axis.set_xticks(positions, ["Original defaults", "Plausible fixed", "Protocol selected", "Grid selected"])
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("ORL gray: 2 train / 2 validation / 6 test per subject\nMean of seeds 10–14; error bars = split standard deviation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/orl_gray_improved")
    run(parser.parse_args().output)
