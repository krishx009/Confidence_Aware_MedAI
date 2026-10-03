"""
evaluate.py

[Base Paper + Extension, Phase 3] Formal evaluation of all uncertainty estimation methods
built across this project, per the project's original evaluation scope:

    "Implement proper evaluation metrics: Area Above the rejection Curve (AAC), Expected
    Calibration Error (ECE), Throwaway Rate at target accuracy (e.g., 99%), AUROC for
    misclassification detection. Compare my adaptive method against: (a) the base paper's
    static CAE, (b) a vanilla ensemble without soft labels, (c) a single model with
    softmax confidence."

AAC and TRA-99 were already implemented and validated in uncertainty.py (Milestone 3) and
uncertainty_adaptive.py (Milestone 4). This script adds ECE and AUROC, adds the single-model
baseline, and produces the final four-way comparison table plus calibration and ROC plots.

[Extension] Baseline definitions used in this comparison:
    - Single model: base ensemble model_0's own softmax confidence, |p - 0.5| distance from
      the decision boundary as the uncertainty score. Model 0 was chosen arbitrarily among
      the 10 available models, since any single one serves equally well as the "no ensemble"
      floor baseline.
    - Vanilla ensemble: TV(Ens, Ens) -- epistemic + aleatoric, both derived from the base
      hard-label ensemble, using no soft-label/expert information at all.
    - Static CAE: TV(Ens, CAE) -- the base paper's main method (Milestone 3).
    - Adaptive CAE: this project's redesigned reliability-weighted method (Milestone 4),
      evaluated at temperature=0.05, the best-validated setting from the Milestone 4 sweep.
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
RESULTS_PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

BASE_SPLIT_SEED = 42
FINAL_ADAPTIVE_TEMPERATURE = 0.05  # [Extension] best setting from Milestone 4 sweep
ECE_NUM_BINS = 10  # [Base Paper, Sec 4.1] matches the paper's own ECE bin count


# ---------------------------------------------------------------------------
# New metrics for Phase 3
# ---------------------------------------------------------------------------

def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_NUM_BINS) -> float:
    """
    [Base Paper, Sec 4.1] Standard ECE: bins predictions by confidence into n_bins
    equal-width bins over [0, 1], and computes the population-weighted average absolute
    gap between each bin's mean confidence and its actual accuracy.

    confidence: (N,) -- the model's own confidence in its prediction, i.e. max(p, 1-p)
        for a binary classifier (NOT the raw class-1 probability).
    correct: (N,) -- boolean/0-1 array, whether the prediction was correct.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)  # include the right edge on the last bin
        else:
            in_bin = (confidence >= lo) & (confidence < hi)

        bin_count = in_bin.sum()
        if bin_count == 0:
            continue

        bin_confidence = confidence[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def auroc_misclassification(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """
    [Extension] AUROC for misclassification detection: treats "was this sample
    misclassified" as the positive class, and the uncertainty score as the detector's
    decision function. A perfect uncertainty measure ranks every misclassified sample
    above every correctly-classified sample, giving AUROC = 1.0.
    """
    misclassified = (~correct.astype(bool)).astype(int)
    if misclassified.sum() == 0 or misclassified.sum() == len(misclassified):
        return float("nan")  # undefined if there are no errors (or no correct predictions) at all
    return float(roc_auc_score(misclassified, uncertainty))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)

    _, _, test_loader, reagent_vocab = get_dataloaders(batch_size=32, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)

    print("Loading base (hard-label) ensemble...")
    base_models = load_ensemble(BASE_CHECKPOINT_DIR, num_reagents, device)
    print("Loading CAE (soft-label) ensemble...")
    cae_models = load_ensemble(CAE_CHECKPOINT_DIR, num_reagents, device)

    print("\nRunning inference on test set with base ensemble...")
    base_predictions, hard_labels = collect_predictions(base_models, test_loader, device)
    print("Running inference on test set with CAE ensemble...")
    cae_predictions, _ = collect_predictions(cae_models, test_loader, device)

    n_test = len(hard_labels)
    print(f"\nTest set size: {n_test}\n")

    # --- Reliability weights for the final adaptive configuration ---
    print("Building reliability reference sets for base ensemble...")
    base_reference_sets = build_all_reference_sets(base_models, BASE_SPLIT_SEED, batch_size=32, device=device)
    print("Extracting test embeddings from base ensemble...")
    base_test_embeddings = extract_test_embeddings(base_models, test_loader, device)
    print(f"Computing reliability weights (k={DEFAULT_K}, temperature={FINAL_ADAPTIVE_TEMPERATURE})...\n")
    weights_base = compute_reliability_weights(
        base_test_embeddings, base_reference_sets, k=DEFAULT_K, temperature=FINAL_ADAPTIVE_TEMPERATURE
    )

    # =========================================================================
    # Build each method's predictions, correctness, confidence, and uncertainty
    # =========================================================================
    methods = {}

    # --- (c) Single model baseline: model_0's own predictions, no ensemble ---
    single_model_probs = base_predictions[0]  # (N,)
    single_model_hard_preds = (single_model_probs >= 0.5).astype(int)
    single_model_correct = (single_model_hard_preds == hard_labels.astype(int))
    single_model_confidence = np.maximum(single_model_probs, 1.0 - single_model_probs)  # max(p, 1-p)
    single_model_uncertainty = 1.0 - single_model_confidence  # equivalently |p - 0.5| distance, inverted
    methods["Single Model"] = {
        "correct": single_model_correct,
        "confidence": single_model_confidence,
        "uncertainty": single_model_uncertainty,
    }

    # --- Shared ensemble-based correctness/confidence (mean base ensemble prediction) ---
    mean_base_pred = base_predictions.mean(axis=0)
    ensemble_hard_preds = (mean_base_pred >= 0.5).astype(int)
    ensemble_correct = (ensemble_hard_preds == hard_labels.astype(int))
    ensemble_confidence = np.maximum(mean_base_pred, 1.0 - mean_base_pred)

    static_epistemic = epistemic_uncertainty(base_predictions)
    static_aleatoric_ens = aleatoric_uncertainty(base_predictions)
    static_aleatoric_cae = aleatoric_uncertainty(cae_predictions)

    # --- (b) Vanilla ensemble: TV(Ens, Ens), no soft-label information ---
    methods["Vanilla Ensemble"] = {
        "correct": ensemble_correct,
        "confidence": ensemble_confidence,
        "uncertainty": static_epistemic + static_aleatoric_ens,
    }

    # --- (a) Static CAE: TV(Ens, CAE), base paper's main method ---
    methods["Static CAE"] = {
        "correct": ensemble_correct,
        "confidence": ensemble_confidence,
        "uncertainty": static_epistemic + static_aleatoric_cae,
    }

    # --- This project's method: Adaptive CAE ---
    adaptive_epistemic = weighted_epistemic_uncertainty(base_predictions, weights_base)
    methods["Adaptive CAE (ours)"] = {
        "correct": ensemble_correct,
        "confidence": ensemble_confidence,
        "uncertainty": adaptive_epistemic + static_aleatoric_cae,
    }

    # =========================================================================
    # Evaluate all four methods across AAC, TRA-99, ECE, AUROC
    # =========================================================================
    thresholds = np.linspace(0.0, 0.99, 100)
    results = {}
    for name, m in methods.items():
        aac = compute_aac(m["uncertainty"], m["correct"], thresholds)
        tra99 = compute_tra_at_target(m["uncertainty"], m["correct"], thresholds, target=0.99)
        ece = expected_calibration_error(m["confidence"], m["correct"], n_bins=ECE_NUM_BINS)
        auroc = auroc_misclassification(m["uncertainty"], m["correct"])
        results[name] = {
            "accuracy": float(m["correct"].mean()),
            "aac_x1e4": float(aac * 1e4),
            "tra99_pct": float(tra99),
            "ece": float(ece),
            "auroc": float(auroc),
        }

    # --- Print final comparison table ---
    print("=" * 100)
    print("PHASE 3: FINAL COMPARISON")
    print("=" * 100)
    header = f"{'Method':<22} | {'Accuracy':>9} | {'AAC (x10^-4)':>13} | {'TRA-99 (%)':>11} | {'ECE':>8} | {'AUROC':>8}"
    print(header)
    print("-" * 100)
    for name in ["Single Model", "Vanilla Ensemble", "Static CAE", "Adaptive CAE (ours)"]:
        r = results[name]
        print(f"{name:<22} | {r['accuracy']:>9.4f} | {r['aac_x1e4']:>13.2f} | "
              f"{r['tra99_pct']:>11.2f} | {r['ece']:>8.4f} | {r['auroc']:>8.4f}")
    print("=" * 100)
    print(f"\nBase paper reports (Sec 5.4): ensemble ECE = 0.0245 (M=10 bins)")

    # =========================================================================
    # Plots
    # =========================================================================

    # --- Rejection curves, all four methods ---
    plt.figure(figsize=(8, 6))
    for name, m in methods.items():
        accs = rejection_curve(m["uncertainty"], m["correct"], thresholds) * 100
        plt.plot(thresholds * 100, accs, label=name)
    plt.xlabel("Rejected data part, %")
    plt.ylabel("Accuracy, %")
    plt.title("Phase 3: Accuracy-Rejection Curves, All Methods")
    plt.legend()
    plt.grid(alpha=0.3)
    rejection_plot_path = os.path.join(RESULTS_PLOTS_DIR, "phase3_rejection_curves.png")
    plt.savefig(rejection_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nRejection curve plot saved to: {rejection_plot_path}")

    # --- Calibration (reliability) diagram ---
    plt.figure(figsize=(7, 7))
    bin_edges = np.linspace(0.0, 1.0, ECE_NUM_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for name, m in methods.items():
        bin_accs = []
        for i in range(ECE_NUM_BINS):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == ECE_NUM_BINS - 1:
                in_bin = (m["confidence"] >= lo) & (m["confidence"] <= hi)
            else:
                in_bin = (m["confidence"] >= lo) & (m["confidence"] < hi)
            bin_accs.append(m["correct"][in_bin].mean() if in_bin.sum() > 0 else np.nan)
        plt.plot(bin_centers, bin_accs, marker="o", label=name)
    plt.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, label="Perfect calibration")
    plt.xlabel("Confidence (binned)")
    plt.ylabel("Accuracy")
    plt.title("Phase 3: Calibration (Reliability) Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    calibration_plot_path = os.path.join(RESULTS_PLOTS_DIR, "phase3_calibration_diagram.png")
    plt.savefig(calibration_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration diagram saved to: {calibration_plot_path}")

    # --- ROC curves for misclassification detection ---
    plt.figure(figsize=(7, 7))
    for name, m in methods.items():
        misclassified = (~m["correct"].astype(bool)).astype(int)
        if misclassified.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(misclassified, m["uncertainty"])
        plt.plot(fpr, tpr, label=f"{name} (AUROC={results[name]['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Phase 3: ROC Curves for Misclassification Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    roc_plot_path = os.path.join(RESULTS_PLOTS_DIR, "phase3_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved to: {roc_plot_path}")

    # --- Save final metrics JSON ---
    import json
    metrics_path = os.path.join(RESULTS_METRICS_DIR, "phase3_final_evaluation.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_test": int(n_test),
            "ece_num_bins": ECE_NUM_BINS,
            "adaptive_temperature": FINAL_ADAPTIVE_TEMPERATURE,
            "results": results,
        }, f, indent=2)
    print(f"Final metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()