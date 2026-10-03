"""
uncertainty.py

[Base Paper, Section 3, Eq. 2-4] Implements the Law of Total Variance decomposition of
predictive uncertainty into epistemic and aleatoric components, using the trained base
(hard-label) and CAE (soft-label fine-tuned) ensembles.

[Base Paper, Section 5.3] Implements rejection-curve evaluation: Area Above the rejection
Curve (AAC), normalized against an oracle curve that rejects all misclassified samples
first, and TRA-99 (throwaway rate required to reach 99% accuracy).

[Base Paper, Section 5.4, Table 1] Reproduces three configurations for direct comparison
against the paper's reported numbers:
    TV(Ens, -)   : epistemic uncertainty only
    TV(Ens, Ens) : epistemic + aleatoric, both estimated from the base ensemble
    TV(Ens, CAE) : epistemic (base ensemble) + aleatoric (CAE ensemble) -- main method
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from utils import MobileNetV3Reagent

# [Extension] NumPy 2.0+ compatibility: np.trapz was removed in favor of np.trapezoid.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
CAE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "cae_ensemble")
RESULTS_PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

NUM_MODELS = 10
BASE_SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Inference: collect per-model predictions on the test set
# ---------------------------------------------------------------------------

def load_ensemble(checkpoint_dir: str, num_reagents: int, device: torch.device):
    """Loads all K checkpoints from a directory into a list of eval-mode models."""
    models = []
    for i in range(NUM_MODELS):
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{i}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def collect_predictions(models, test_loader, device: torch.device):
    """
    Runs all K models over the full test set.
    Returns:
        predictions: np.ndarray, shape (K, N) -- per-model probability predictions
        hard_labels: np.ndarray, shape (N,) -- ground-truth hard labels
    """
    all_predictions = [[] for _ in models]
    hard_labels_list = []

    with torch.no_grad():
        for images, reagents, hard_labels, _soft_labels in test_loader:
            images, reagents = images.to(device), reagents.to(device)
            hard_labels_list.append(hard_labels.numpy())

            for k, model in enumerate(models):
                logits = model(images, reagents)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_predictions[k].append(probs)

    predictions = np.stack([np.concatenate(p) for p in all_predictions], axis=0)  # (K, N)
    hard_labels = np.concatenate(hard_labels_list)  # (N,)
    return predictions, hard_labels


# ---------------------------------------------------------------------------
# Uncertainty decomposition [Base Paper, Eq. 3-4]
# ---------------------------------------------------------------------------

def epistemic_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """[Eq. 3] predictions: (K, N) -> returns (N,) variance across the K models."""
    mean_pred = predictions.mean(axis=0, keepdims=True)  # (1, N)
    return np.mean((predictions - mean_pred) ** 2, axis=0)  # (N,)


def aleatoric_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """[Eq. 4] predictions: (K, N) -> returns (N,) mean of p*(1-p) across the K models."""
    return np.mean(predictions * (1.0 - predictions), axis=0)  # (N,)


# ---------------------------------------------------------------------------
# Rejection curves, AAC, TRA-99 [Base Paper, Section 5.3]
# ---------------------------------------------------------------------------

def rejection_curve(uncertainty: np.ndarray, correct: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    For each rejection fraction t in thresholds, rejects the t*N samples with highest
    uncertainty and computes accuracy on the remainder.
    uncertainty: (N,), correct: (N,) boolean/0-1 array. Returns accuracy per threshold.
    """
    n = len(uncertainty)
    order = np.argsort(-uncertainty)  # descending: most uncertain first
    correct_sorted = correct[order]

    accuracies = []
    for t in thresholds:
        num_reject = int(round(t * n))
        remaining = correct_sorted[num_reject:]
        if len(remaining) == 0:
            accuracies.append(1.0)  # nothing left to be wrong about
        else:
            accuracies.append(remaining.mean())
    return np.array(accuracies)


def compute_aac(uncertainty: np.ndarray, correct: np.ndarray, thresholds: np.ndarray) -> float:
    """
    [Base Paper, Section 5.3] Area Above the rejection Curve, normalized by subtracting
    the area above the oracle curve (which rejects misclassified samples first).
    Returns AAC as a raw fraction (multiply by 1e4 to match the paper's x10^-4 units).
    """
    method_acc = rejection_curve(uncertainty, correct, thresholds)
    method_aac = _trapezoid(1.0 - method_acc, thresholds)

    # Oracle: assign highest "uncertainty" to misclassified samples, so they're rejected first.
    oracle_uncertainty = (~correct.astype(bool)).astype(float)
    oracle_acc = rejection_curve(oracle_uncertainty, correct, thresholds)
    oracle_aac = _trapezoid(1.0 - oracle_acc, thresholds)

    return method_aac - oracle_aac


def compute_tra_at_target(uncertainty: np.ndarray, correct: np.ndarray, thresholds: np.ndarray, target: float = 0.99) -> float:
    """Smallest rejection fraction (%) at which accuracy reaches `target`, via interpolation."""
    accs = rejection_curve(uncertainty, correct, thresholds)
    for i, acc in enumerate(accs):
        if acc >= target:
            return thresholds[i] * 100.0
    return 100.0  # never reached target accuracy even at full rejection


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

    print("Running inference on test set with base ensemble...")
    base_predictions, hard_labels = collect_predictions(base_models, test_loader, device)
    print("Running inference on test set with CAE ensemble...")
    cae_predictions, _ = collect_predictions(cae_models, test_loader, device)

    n_test = len(hard_labels)
    print(f"\nTest set size: {n_test}")

    # Final classification prediction: mean of base ensemble, thresholded at 0.5.
    mean_base_pred = base_predictions.mean(axis=0)
    hard_preds = (mean_base_pred >= 0.5).astype(int)
    correct = (hard_preds == hard_labels.astype(int))
    accuracy = correct.mean()
    print(f"Base ensemble test accuracy (mean prediction, threshold 0.5): {accuracy:.4f}\n")

    # Uncertainty components
    epistemic = epistemic_uncertainty(base_predictions)              # from base ensemble
    aleatoric_from_ens = aleatoric_uncertainty(base_predictions)      # baseline: base ensemble again
    aleatoric_from_cae = aleatoric_uncertainty(cae_predictions)       # main method: CAE ensemble

    tv_ens_only = epistemic
    tv_ens_ens = epistemic + aleatoric_from_ens
    tv_ens_cae = epistemic + aleatoric_from_cae

    # --- Diagnostic: inspect the raw distributions before trusting downstream metrics ---
    print("=" * 70)
    print("Diagnostic: uncertainty component statistics")
    print("-" * 70)
    for name, arr in [
        ("epistemic (base ensemble)", epistemic),
        ("aleatoric from base ensemble", aleatoric_from_ens),
        ("aleatoric from CAE ensemble", aleatoric_from_cae),
    ]:
        print(f"{name:<32} mean={arr.mean():.6f}  std={arr.std():.6f}  "
              f"min={arr.min():.6f}  max={arr.max():.6f}")

    corr_ens = np.corrcoef(epistemic, aleatoric_from_ens)[0, 1]
    corr_cae = np.corrcoef(epistemic, aleatoric_from_cae)[0, 1]
    print(f"\nCorrelation(epistemic, aleatoric_from_ens) = {corr_ens:.4f}")
    print(f"Correlation(epistemic, aleatoric_from_cae) = {corr_cae:.4f}")

    # How many samples have near-zero aleatoric signal (overconfident models)?
    near_zero_ens = (aleatoric_from_ens < 0.01).mean() * 100
    near_zero_cae = (aleatoric_from_cae < 0.01).mean() * 100
    print(f"\n% samples with aleatoric_from_ens < 0.01: {near_zero_ens:.1f}%")
    print(f"% samples with aleatoric_from_cae < 0.01: {near_zero_cae:.1f}%")
    print("=" * 70)
    print()

    # Rejection-curve evaluation
    thresholds = np.linspace(0.0, 0.99, 100)  # 0% to 99% rejection, matching paper's plots

    results = {}
    for name, unc in [("Ens, -", tv_ens_only), ("Ens, Ens", tv_ens_ens), ("Ens, CAE", tv_ens_cae)]:
        aac = compute_aac(unc, correct, thresholds)
        tra99 = compute_tra_at_target(unc, correct, thresholds, target=0.99)
        results[name] = {"aac": aac, "aac_x1e4": aac * 1e4, "tra99_pct": tra99}

    # --- Print comparison table ---
    print("=" * 70)
    print(f"{'Method':<12} | {'AAC (x10^-4)':>14} | {'TRA-99 (%)':>12}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<12} | {r['aac_x1e4']:>14.2f} | {r['tra99_pct']:>12.2f}")
    print("=" * 70)
    print("\nBase paper reports (Table 1):")
    print(f"{'Ens, -':<12} | {50.0:>14.2f} | {15.4:>12.2f}")
    print(f"{'Ens, Ens':<12} | {43.6:>14.2f} | {11.8:>12.2f}")
    print(f"{'Ens, CAE':<12} | {21.4:>14.2f} | {9.2:>12.2f}")

    # --- Save rejection curve plot (Fig. 5 style) ---
    plt.figure(figsize=(8, 6))
    for name, unc in [("Ens, -", tv_ens_only), ("Ens, Ens", tv_ens_ens), ("Ens, CAE", tv_ens_cae)]:
        accs = rejection_curve(unc, correct, thresholds) * 100
        plt.plot(thresholds * 100, accs, label=name)
    plt.xlabel("Rejected data part, %")
    plt.ylabel("Accuracy, %")
    plt.title("Accuracy-Rejection Curves (Classification)")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(RESULTS_PLOTS_DIR, "rejection_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nRejection curve plot saved to: {plot_path}")

    # --- Save metrics as JSON ---
    import json
    metrics_path = os.path.join(RESULTS_METRICS_DIR, "uncertainty_results.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": float(accuracy),
            "n_test": int(n_test),
            "results": results,
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()