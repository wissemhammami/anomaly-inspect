import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

THRESHOLD_PERCENTILE = 95
SPLIT_SEED = 42
FIT_FRACTION = 0.8  # fraction of train "good" used to fit the Gaussian

EMB_PATHS = {
    "frozen": "models/resnet18_frozen_embeddings.pt",
    "finetuned": "models/resnet18_finetuned_embeddings.pt",
}


def fit_gaussian(train_emb):
    mean = train_emb.mean(dim=0).numpy()
    cov = np.cov(train_emb.numpy(), rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    return mean, cov_inv


def mahalanobis_scores(emb, mean, cov_inv):
    diff = emb.numpy() - mean
    return np.einsum("ij,jk,ik->i", diff, cov_inv, diff)


def evaluate(path):
    data = torch.load(path)
    train_emb, test_emb = data["train_emb"], data["test_emb"]
    test_labels = data["test_labels"].numpy()

    rng = np.random.RandomState(SPLIT_SEED)
    n = train_emb.shape[0]
    perm = rng.permutation(n)
    split = int(n * FIT_FRACTION)
    fit_idx, thresh_idx = perm[:split], perm[split:]

    mean, cov_inv = fit_gaussian(train_emb[fit_idx])

    thresh_scores = mahalanobis_scores(train_emb[thresh_idx], mean, cov_inv)
    thresh = float(np.percentile(thresh_scores, THRESHOLD_PERCENTILE))

    test_scores = mahalanobis_scores(test_emb, mean, cov_inv)
    auc = roc_auc_score(test_labels, test_scores)
    preds = (test_scores >= thresh).astype(int)
    acc = (preds == test_labels).mean()

    return {
        "roc_auc": round(float(auc), 4),
        "threshold": round(thresh, 4),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "accuracy": round(float(acc), 4),
        "mean_score_good": round(float(test_scores[test_labels == 0].mean()), 2),
        "mean_score_defect": round(float(test_scores[test_labels == 1].mean()), 2),
        "gaussian_fit_size": int(len(fit_idx)),
        "threshold_calibration_size": int(len(thresh_idx)),
        "test_size": int(len(test_scores)),
    }


def main():
    results = {}
    for name, path in EMB_PATHS.items():
        results[name] = evaluate(path)
        print(f"\n{name}")
        for k, v in results[name].items():
            print(f"  {k}: {v}")

    with open("results/anomaly_detection_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()