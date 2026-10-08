"""Independent checks of update order, tensor axes, numerical properties, and the recognition pipeline."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from csa import CSA, PCA, LDA, fold, unfold, mode_product, nearest_neighbor, reconstruction_metrics
from run_orl_gray import split_orl


def check():
    rng = np.random.default_rng(314)
    passed = []
    x = rng.normal(size=(12, 5, 7))
    u, v = np.linalg.qr(rng.normal(size=(5, 2)))[0], np.linalg.qr(rng.normal(size=(7, 3)))[0]
    got = mode_product(mode_product(x, u.T, 0), v.T, 1)
    expected = np.stack([u.T @ sample @ v for sample in x])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    tensor = rng.normal(size=(9, 5, 7, 4))
    for mode in range(3):
        np.testing.assert_array_equal(fold(unfold(tensor, mode), tensor.shape, mode), tensor)
    passed.append("non-square tensor axes and unfold/fold round-trip")

    # Independent 2D matrix updates: compare NumPy eigh with the main SciPy solver.
    reference_u, reference_v = np.eye(5, 2), np.eye(7, 3)
    cu = sum(sample @ reference_v @ reference_v.T @ sample.T for sample in x)
    reference_u = np.linalg.eigh(cu)[1][:, -2:]
    cv = sum(sample.T @ reference_u @ reference_u.T @ sample for sample in x)
    reference_v = np.linalg.eigh(cv)[1][:, -3:]
    fitted = CSA((2, 3), max_iter=1).fit(x)
    for actual, expected in zip(fitted.bases, [reference_u, reference_v]):
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-11)
    passed.append("2-D sequential update agrees with independent matrix formulas")

    # Independent per-sample 3D einsum contractions; do not reuse mode_product for self-verification.
    ranks = (2, 3, 2)
    reference = [np.eye(m, r) for m, r in zip(tensor.shape[1:], ranks)]
    for k in range(3):
        covariance = np.zeros((tensor.shape[k + 1],) * 2)
        for sample in tensor:
            if k == 0:
                intermediate = np.einsum("abc,bj,ck->ajk", sample, reference[1], reference[2])
            elif k == 1:
                intermediate = np.moveaxis(np.einsum("abc,ai,ck->ibk", sample, reference[0], reference[2]), 1, 0)
            else:
                intermediate = np.moveaxis(np.einsum("abc,ai,bj->ijc", sample, reference[0], reference[1]), 2, 0)
            b = intermediate.reshape(tensor.shape[k + 1], -1)
            covariance += b @ b.T
        reference[k] = np.linalg.eigh(covariance)[1][:, -ranks[k]:]
    fitted3 = CSA(ranks, max_iter=1).fit(tensor)
    for actual, expected in zip(fitted3.bases, reference):
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-10)
    direct_core = np.einsum("nabc,ai,bj,ck->nijk", tensor, *fitted3.bases)
    np.testing.assert_allclose(fitted3.transform(tensor), direct_core, atol=1e-11)
    passed.append("3-D sequential update and shared-core transform agree with independent einsum")

    for center in [False, True]:
        fitted = CSA((2, 2), center=center, init="random", max_iter=8).fit(x)
        sequence = [fitted.initial_sse] + [value for h in fitted.history for value in h["mode_sse"]]
        assert np.max(np.diff(sequence)) < 1e-9
        for basis in fitted.bases:
            np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
        reconstructed = fitted.inverse_transform(fitted.transform(x))
        np.testing.assert_allclose(np.sum((x - reconstructed) ** 2), fitted.history[-1]["sse"], atol=1e-10)
        pca = PCA(4, center=center).fit(x)
        assert np.sum((x - pca.inverse_transform(pca.transform(x))) ** 2) <= sequence[-1] + 1e-9
        full = CSA(tensor.shape[1:], center=center, max_iter=1).fit(tensor)
        np.testing.assert_allclose(full.inverse_transform(full.transform(tensor)), tensor, atol=1e-11)
        matrix = x.reshape(12, -1)
        first_order = CSA((4,), center=center).fit(matrix)
        vector_pca = PCA(4, center=center).fit(matrix)
        np.testing.assert_allclose(first_order.bases[0] @ first_order.bases[0].T,
                                   vector_pca.basis @ vector_pca.basis.T, atol=1e-11)
    passed.append("orthogonality, per-mode monotonic SSE, full reconstruction, first-order PCA equivalence, PCA optimum; both mean settings")

    features = rng.normal(size=(36, 5))
    labels = np.repeat(np.arange(4), 9)
    features += labels[:, None] * np.array([2, -1, 0.5, 0, 1])
    lda = LDA().fit(features, labels)
    np.testing.assert_allclose(lda.basis.T @ lda.sw @ lda.basis, np.eye(3), atol=1e-11)
    eigenvalues = eigh(lda.sb, lda.sw, eigvals_only=True)[::-1][:3]
    np.testing.assert_allclose(lda.eigenvalues, eigenvalues, atol=1e-11)
    np.testing.assert_allclose(lda.sb @ lda.basis, lda.sw @ lda.basis @ np.diag(lda.eigenvalues), atol=1e-9)
    # Enforce the pre-LDA N-Nc dimension limit; do not silently regularize invalid cases.
    try:
        LDA().fit(np.ones((4, 3)), np.array([0, 0, 1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("LDA dimension restriction not enforced")
    prediction, indices, distance = nearest_neighbor(np.array([[0., 0.], [3., 4.]]),
                                                    np.array([7, 9]), np.array([[0., 1.], [3., 5.]]), block_size=1)
    np.testing.assert_array_equal(prediction, [7, 9])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(distance, [1, 1])
    passed.append("LDA matches generalized eigenproblem; dimension guard; 1-NN hand example")

    zeros = np.zeros((2, 3, 4, 5))
    metric = reconstruction_metrics(zeros, np.ones_like(zeros), method="csa", dimensions=(2, 2, 2))
    assert metric["sse"] == 120 and metric["stored_scalars"] == 40 and metric["compression_ratio"] == 3
    np.testing.assert_allclose(metric["paper_rmse"], np.sqrt(60))
    mean_metric = reconstruction_metrics(zeros, zeros, method="pca", dimensions=2, include_mean=True)
    assert mean_metric["stored_scalars"] == 184
    passed.append("paper RMSE and total-storage CR verified against hand calculation")

    orl_labels = np.repeat(np.arange(1, 41), 10)
    for seed in [0, 1, 2]:
        train, probe = split_orl(orl_labels, seed)
        assert all(np.sum(orl_labels[train] == c) == 2 and np.sum(orl_labels[probe] == c) == 8 for c in range(1, 41))
        np.testing.assert_array_equal(split_orl(orl_labels, seed)[0], train)
    passed.append("ORL per-class 2/8 split, coverage, isolation and deterministic replay")
    return {"status": "passed", "seed": 314, "checks": passed}


if __name__ == "__main__":
    result = check()
    target = Path(__file__).resolve().parent / "results/numerical_checks.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
