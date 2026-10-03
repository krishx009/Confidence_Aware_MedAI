"""
evaluate_expert_reliability.py

[Extension, Phase 2b] Final comparison across all four uncertainty configurations this
project has built, evaluated identically on the full 2,516-image test set:

    1. Static CAE              -- base paper's method: uniform model weights,
                                   uniform (mean-of-6) expert soft labels.
    2. Adaptive CAE             -- Phase 2: model-level reliability weighting
                                   (epistemic term only), static expert soft labels.
    3. Expert-Reliability CAE   -- Phase 2b: uniform model weights, but the CAE
                                   ensemble is fine-tuned on reliability-weighted
                                   expert soft labels (Path B, full retrain).
    4. Fully Combined           -- Phase 2b: model-level reliability weighting AND
                                   the expert-reliability-trained CAE ensemble together
                                   -- the complete "Adaptive Reliability-Aware" system
                                   matching the project's abstract in full.

All four share the same base (hard-label) ensemble for epistemic uncertainty; they
differ only in which CAE ensemble supplies the aleatoric term and whether model-level
reliability weighting is applied to the epistemic term.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from uncertainty import (
    load_ensemble,
    collect_predictions,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    rejection_curve,
    compute_aac,
    compute_tra_at_target,
)
from reliability import (
    build_all_reference_sets,
    extract_test_embeddings,
    compute_reliability_weights,
    weighted_epistemic_uncertainty,
    DEFAULT_K,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
CAE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "cae_ensemble")
CAE_EXPERT_REL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "cae_ensemble_expert_reliability")
RESULTS_PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

BASE_SPLIT_SEED = 42
ADAPTIVE_TEMPERATURE = 0.05  # best setting from the Phase 2 temperature sweep
ECE_NUM_BINS = 10


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_NUM_BINS) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        ece += (bin_count / n) * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def auroc_misclassification(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    misclassified = (~correct.astype(bool)).astype(int)
    if misclassified.sum() == 0 or misclassified.sum() == len(misclassified):
        return float("nan")
    return float(roc_auc_score(misclassified, uncertainty))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)

    _, _, test_loader, reagent_vocab = get_dataloaders(batch_size=32, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)

    print("Loading base (hard-label) ensemble...")
    base_models = load_ensemble(BASE_CHECKPOINT_DIR, num_reagents, device)
    print("Loading static CAE ensemble...")
    cae_static_models = load_ensemble(CAE_CHECKPOINT_DIR, num_reagents, device)
    print("Loading expert-reliability CAE ensemble...")
    cae_expert_rel_models = load_ensemble(CAE_EXPERT_REL_CHECKPOINT_DIR, num_reagents, device)

    print("\nRunning inference on test set with base ensemble...")
    base_predictions, hard_labels = collect_predictions(base_models, test_loader, device)
    print("Running inference on test set with static CAE ensemble...")
    cae_static_predictions, _ = collect_predictions(cae_static_models, test_loader, device)
    print("Running inference on test set with expert-reliability CAE ensemble...")
    cae_expert_rel_predictions, _ = collect_predictions(cae_expert_rel_models, test_loader, device)

    n_test = len(hard_labels)
    mean_base_pred = base_predictions.mean(axis=0)
    hard_preds = (mean_base_pred >= 0.5).astype(int)
    correct = (hard_preds == hard_labels.astype(int))
    confidence = np.maximum(mean_base_pred, 1.0 - mean_base_pred)
    print(f"\nTest set size: {n_test}")
    print(f"Base ensemble test accuracy: {correct.mean():.4f}\n")

    # --- Model-level reliability weights (Phase 2), reused for both adaptive configs ---
    print("Building model-level reliability reference sets...")
    base_reference_sets = build_all_reference_sets(base_models, BASE_SPLIT_SEED, batch_size=32, device=device)
    print("Extracting test embeddings for model-level reliability...")
    base_test_embeddings = extract_test_embeddings(base_models, test_loader, device)
    print(f"Computing model-level reliability weights (k={DEFAULT_K}, T={ADAPTIVE_TEMPERATURE})...\n")
    weights_base = compute_reliability_weights(
        base_test_embeddings, base_reference_sets, k=DEFAULT_K, temperature=ADAPTIVE_TEMPERATURE
    )

    # --- Uncertainty components ---
    static_epistemic = epistemic_uncertainty(base_predictions)
    adaptive_epistemic = weighted_epistemic_uncertainty(base_predictions, weights_base)
    aleatoric_static_cae = aleatoric_uncertainty(cae_static_predictions)
    aleatoric_expert_rel_cae = aleatoric_uncertainty(cae_expert_rel_predictions)

    configs = {
        "Static CAE": static_epistemic + aleatoric_static_cae,
        "Adaptive CAE (model-level)": adaptive_epistemic + aleatoric_static_cae,
        "Expert-Reliability CAE": static_epistemic + aleatoric_expert_rel_cae,
        "Fully Combined": adaptive_epistemic + aleatoric_expert_rel_cae,
    }

    # --- Evaluate all four ---
    thresholds = np.linspace(0.0, 0.99, 100)
    results = {}
    for name, unc in configs.items():
        aac = compute_aac(unc, correct, thresholds)
        tra99 = compute_tra_at_target(unc, correct, thresholds, target=0.99)
        ece = expected_calibration_error(confidence, correct, n_bins=ECE_NUM_BINS)  # shared point estimate
        auroc = auroc_misclassification(unc, correct)
        results[name] = {
            "aac_x1e4": float(aac * 1e4),
            "tra99_pct": float(tra99),
            "ece": float(ece),
            "auroc": float(auroc),
        }

    print("=" * 100)
    print("FINAL COMPARISON \u2014 All Four Configurations")
    print("=" * 100)
    header = f"{'Configuration':<28} | {'AAC (x10^-4)':>13} | {'TRA-99 (%)':>11} | {'ECE':>8} | {'AUROC':>8}"
    print(header)
    print("-" * 100)
    for name in ["Static CAE", "Adaptive CAE (model-level)", "Expert-Reliability CAE", "Fully Combined"]:
        r = results[name]
        print(f"{name:<28} | {r['aac_x1e4']:>13.2f} | {r['tra99_pct']:>11.2f} | "
              f"{r['ece']:>8.4f} | {r['auroc']:>8.4f}")
    print("=" * 100)

    static_aac = results["Static CAE"]["aac_x1e4"]
    print("\nAAC improvement vs. Static CAE baseline:")
    for name in ["Adaptive CAE (model-level)", "Expert-Reliability CAE", "Fully Combined"]:
        delta_pct = (results[name]["aac_x1e4"] - static_aac) / static_aac * 100
        print(f"  {name:<28}: {delta_pct:+.1f}%")

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    for name, unc in configs.items():
        accs = rejection_curve(unc, correct, thresholds) * 100
        plt.plot(thresholds * 100, accs, label=name)
    plt.xlabel("Rejected data part, %")
    plt.ylabel("Accuracy, %")
    plt.title("Accuracy-Rejection Curves: All Four Configurations")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(RESULTS_PLOTS_DIR, "expert_reliability_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    # --- ROC curves ---
    plt.figure(figsize=(7, 7))
    for name, unc in configs.items():
        misclassified = (~correct.astype(bool)).astype(int)
        fpr, tpr, _ = roc_curve(misclassified, unc)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={results[name]['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Misclassification Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    roc_path = os.path.join(RESULTS_PLOTS_DIR, "expert_reliability_roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"ROC plot saved to: {roc_path}")

    # --- Save metrics JSON ---
    import json
    metrics_path = os.path.join(RESULTS_METRICS_DIR, "expert_reliability_final_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_test": int(n_test),
            "base_ensemble_accuracy": float(correct.mean()),
            "adaptive_temperature": ADAPTIVE_TEMPERATURE,
            "k_neighbors": DEFAULT_K,
            "results": results,
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()