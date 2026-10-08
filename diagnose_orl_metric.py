"""固定划分与维数，只对照 LDA 方向缩放和 NN 距离，不选择最佳参数。

所有候选完整输出。不是新的主复现协议，也不替换原实验结果。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from csa import CSA, PCA, LDA
from run_orl_gray import load_orl, split_orl


def diagnose(output):
    base = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError("Use a new output name to preserve previous diagnostics")
    x, labels, _, raw_hash = load_orl(base / "data/raw/orl")
    rows = []
    for seed in [0, 1, 2]:
        train, probe = split_orl(labels, seed)
        for center in [False, True]:
            for method, model, dimension in [
                ("PCA", PCA(30, center=center), 18),
                ("CSA", CSA((11, 2), center=center), 16),
            ]:
                model.fit(x[train])
                gallery = model.transform(x[train]).reshape(80, -1)
                probes = model.transform(x[probe]).reshape(320, -1)
                lda = LDA().fit(gallery, labels[train])
                original = lda.basis[:, :dimension]
                for scaling in ["sw_identity", "unit_columns"]:
                    basis = original if scaling == "sw_identity" else original / np.linalg.norm(original, axis=0)
                    # 缩放保留同一特征方向和广义特征方程，改变的是距离权重。
                    np.testing.assert_allclose(
                        lda.sb @ basis, lda.sw @ basis @ np.diag(lda.eigenvalues[:dimension]),
                        rtol=1e-7, atol=1e-5,
                    )
                    g = (gallery - lda.mean) @ basis
                    q = (probes - lda.mean) @ basis
                    for metric in ["euclidean", "cosine"]:
                        distances = cdist(q, g, metric)
                        if not np.isfinite(distances).all():
                            raise ValueError("Nonfinite distance, possibly a zero cosine vector")
                        predictions = labels[train][np.argmin(distances, axis=1)]
                        rows.append({"seed": seed, "center": center, "method": method,
                                     "scaling": scaling, "metric": metric,
                                     "accuracy": float(100 * np.mean(predictions == labels[probe]))})
    summary = []
    for center in [False, True]:
        for method in ["PCA", "CSA"]:
            for scaling in ["sw_identity", "unit_columns"]:
                for metric in ["euclidean", "cosine"]:
                    values = [r["accuracy"] for r in rows if r["center"] == center and r["method"] == method
                              and r["scaling"] == scaling and r["metric"] == metric]
                    summary.append({"center": center, "method": method, "scaling": scaling,
                                    "metric": metric, "mean_percent": float(np.mean(values)),
                                    "sample_std_pp": float(np.std(values, ddof=1))})
    result = {
        "scope": "paired protocol sensitivity; fixed paper dimensions; no parameter selection; original models untouched",
        "split_seeds": [0, 1, 2], "cosine_origin": "subtract LDA training mean",
        "scaling_definitions": {"sw_identity": "W.T Sw W=I; original implementation",
                                "unit_columns": "each LDA direction divided by its Euclidean norm"},
        "limitations": "These probe scores are diagnostic, not an independent test of a selected protocol; no evidence of the author's actual metric.",
        "raw_data_sha256": raw_hash,
        "code_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                        for name in ["csa.py", "run_orl_gray.py", "diagnose_orl_metric.py"]},
        "summary": summary, "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/orl_metric_diagnosis_verified.json")
    diagnose(parser.parse_args().output)
