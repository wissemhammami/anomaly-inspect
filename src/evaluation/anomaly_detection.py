import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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


def best_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def evaluate(path):
    data = torch.load(path)
    train_emb, test_emb = data["train_emb"], data["test_emb"]
    test_labels = data["test_labels"].numpy()

    mean, cov_inv = fit_gaussian(train_emb)
    scores = mahalanobis_scores(test_emb, mean, cov_inv)

    auc = roc_auc_score(test_labels, scores)
    thresh = best_threshold(test_labels, scores)
    preds = (scores >= thresh).astype(int)
    acc = (preds == test_labels).mean()

    return {
        "roc_auc": round(float(auc), 4),
        "threshold": round(float(thresh), 4),
        "accuracy": round(float(acc), 4),
        "mean_score_good": round(float(scores[test_labels == 0].mean()), 2),
        "mean_score_defect": round(float(scores[test_labels == 1].mean()), 2),
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