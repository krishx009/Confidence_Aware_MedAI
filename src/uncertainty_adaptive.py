"""
uncertainty_adaptive.py

[Extension, Phase 2] The project's novel contribution: Adaptive Reliability-Aware
Confidence Learning. Combines the reliability weighting scheme (reliability.py) with the
base paper's uncertainty decomposition (uncertainty.py) to produce reliability-weighted
analogs of TV(Ens,-), TV(Ens,Ens), and TV(Ens,CAE), and compares them directly against
the static (uniform-weighted) baselines established and validated in Milestone 3.

This script is kept separate from uncertainty.py so that Milestone 3's exact validated
results remain independently reproducible without any risk of this extension's code
changing that behavior.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

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

    # --- Predictions (same as uncertainty.py) ---
    print("\nRunning inference on test set with base ensemble...")
    base_predictions, hard_labels = collect_predictions(base_models, test_loader, device)
    print("Running inference on test set with CAE ensemble...")
    cae_predictions, _ = collect_predictions(cae_models, test_loader, device)

    mean_base_pred = base_predictions.mean(axis=0)
    hard_preds = (mean_base_pred >= 0.5).astype(int)
    correct = (hard_preds == hard_labels.astype(int))
    accuracy = correct.mean()
    n_test = len(hard_labels)
    print(f"\nTest set size: {n_test}")
    print(f"Base ensemble test accuracy: {accuracy:.4f}\n")

    # --- Static (uniform-weighted) baselines, recomputed here for a side-by-side table ---
    static_epistemic = epistemic_uncertainty(base_predictions)
    static_aleatoric_ens = aleatoric_uncertainty(base_predictions)
    static_aleatoric_cae = aleatoric_uncertainty(cae_predictions)

    static_tv = {
        "Ens, -": static_epistemic,
        "Ens, Ens": static_epistemic + static_aleatoric_ens,
        "Ens, CAE": static_epistemic + static_aleatoric_cae,
    }

    # --- Reliability weights: built for the base ensemble only. The CAE ensemble's own
    # reliability weights are no longer needed, since aleatoric terms use their validated
    # static (Milestone 3) form -- see note below. ---
    print("Building reliability reference sets for base ensemble...")
    base_reference_sets = build_all_reference_sets(base_models, BASE_SPLIT_SEED, batch_size=32, device=device)

    print("Extracting test embeddings from base ensemble...")
    base_test_embeddings = extract_test_embeddings(base_models, test_loader, device)

    print(f"Computing reliability weights across a temperature sweep (k={DEFAULT_K})...")
    temperatures = [1.0, 0.5, 0.25, 0.1, 0.05]
    weights_by_temp = {
        t: compute_reliability_weights(base_test_embeddings, base_reference_sets, k=DEFAULT_K, temperature=t)
        for t in temperatures
    }

    # --- Diagnostic (reported once, at T=0.5, the previously-chosen default) ---
    max_weight_base = weights_by_temp[0.5].max(axis=0)
    print("\n" + "=" * 78)
    print("Diagnostic: does reliability weighting suppress the uncertainty signal? (T=0.5)")
    print("-" * 78)

    # --- Adaptive (reliability-weighted) uncertainty components, one set per temperature ---
    # [Extension, redesigned] adaptive_epistemic uses reliability weights only for
    # mean-centering, with unweighted (1/K) summation -- see reliability.py docstring.
    # Aleatoric terms are kept at their validated static (Milestone 3) form.
    adaptive_epistemic_by_temp = {
        t: weighted_epistemic_uncertainty(base_predictions, w) for t, w in weights_by_temp.items()
    }

    epistemic_drop = static_epistemic - adaptive_epistemic_by_temp[0.5]
    corr_concentration_vs_drop = np.corrcoef(max_weight_base, epistemic_drop)[0, 1]
    corr_static_unc_vs_drop = np.corrcoef(static_epistemic, epistemic_drop)[0, 1]
    print(f"Correlation(weight concentration, epistemic suppression) = {corr_concentration_vs_drop:.4f}")
    print(f"Correlation(static epistemic uncertainty, suppression amount) = {corr_static_unc_vs_drop:.4f}")
    print(f"Mean static epistemic uncertainty:              {static_epistemic.mean():.6f}")
    print(f"Mean adaptive epistemic uncertainty (T=0.5):     {adaptive_epistemic_by_temp[0.5].mean():.6f}")
    print("=" * 78 + "\n")

    # --- Evaluate static baselines + adaptive configuration at every temperature ---
    thresholds = np.linspace(0.0, 0.99, 100)

    results = {}
    for name, unc in static_tv.items():
        aac = compute_aac(unc, correct, thresholds)
        tra99 = compute_tra_at_target(unc, correct, thresholds, target=0.99)
        results[name] = {"aac": aac, "aac_x1e4": aac * 1e4, "tra99_pct": tra99}

    adaptive_results_by_temp = {}
    for t in temperatures:
        adaptive_epistemic = adaptive_epistemic_by_temp[t]
        configs = {
            "Adaptive Ens, -": adaptive_epistemic,
            "Adaptive Ens, Ens": adaptive_epistemic + static_aleatoric_ens,
            "Adaptive Ens, CAE": adaptive_epistemic + static_aleatoric_cae,
        }
        temp_results = {}
        for name, unc in configs.items():
            aac = compute_aac(unc, correct, thresholds)
            tra99 = compute_tra_at_target(unc, correct, thresholds, target=0.99)
            temp_results[name] = {"aac": aac, "aac_x1e4": aac * 1e4, "tra99_pct": tra99}
        adaptive_results_by_temp[t] = temp_results

    # --- Print static baseline table ---
    print("=" * 78)
    print("Static baselines (Milestone 3, uniform 1/K weighting):")
    print(f"{'Method':<20} | {'AAC (x10^-4)':>14} | {'TRA-99 (%)':>12}")
    print("-" * 78)
    for name in ["Ens, -", "Ens, Ens", "Ens, CAE"]:
        r = results[name]
        print(f"{name:<20} | {r['aac_x1e4']:>14.2f} | {r['tra99_pct']:>12.2f}")
    print("=" * 78)

    # --- Print full temperature sensitivity sweep ---
    print("\nAdaptive (reliability-weighted) results across temperature sweep:")
    print(f"{'Temperature':<12} | {'Method':<20} | {'AAC (x10^-4)':>14} | {'AAC vs static':>14} | {'TRA-99 (%)':>12}")
    print("-" * 90)
    for t in temperatures:
        for base_name, adaptive_name in [
            ("Ens, -", "Adaptive Ens, -"),
            ("Ens, Ens", "Adaptive Ens, Ens"),
            ("Ens, CAE", "Adaptive Ens, CAE"),
        ]:
            r = adaptive_results_by_temp[t][adaptive_name]
            static_aac = results[base_name]["aac_x1e4"]
            delta_pct = ((r["aac_x1e4"] - static_aac) / static_aac) * 100
            print(f"{t:<12} | {adaptive_name:<20} | {r['aac_x1e4']:>14.2f} | {delta_pct:>+13.1f}% | {r['tra99_pct']:>12.2f}")
        print("-" * 90)

    # --- Save rejection curve plot: static (dashed) vs adaptive at T=0.5 (solid) ---
    plot_temperature = 0.5
    adaptive_epistemic_plot = adaptive_epistemic_by_temp[plot_temperature]
    plot_configs = {
        "Ens, -": static_tv["Ens, -"],
        "Ens, Ens": static_tv["Ens, Ens"],
        "Ens, CAE": static_tv["Ens, CAE"],
        "Adaptive Ens, -": adaptive_epistemic_plot,
        "Adaptive Ens, Ens": adaptive_epistemic_plot + static_aleatoric_ens,
        "Adaptive Ens, CAE": adaptive_epistemic_plot + static_aleatoric_cae,
    }

    plt.figure(figsize=(9, 7))
    colors = {"Ens, -": "tab:blue", "Ens, Ens": "tab:orange", "Ens, CAE": "tab:green"}
    for base_name in ["Ens, -", "Ens, Ens", "Ens, CAE"]:
        adaptive_name = f"Adaptive {base_name}"
        static_accs = rejection_curve(plot_configs[base_name], correct, thresholds) * 100
        adaptive_accs = rejection_curve(plot_configs[adaptive_name], correct, thresholds) * 100
        plt.plot(thresholds * 100, static_accs, "--", color=colors[base_name], label=base_name, alpha=0.6)
        plt.plot(thresholds * 100, adaptive_accs, "-", color=colors[base_name], label=adaptive_name)
    plt.xlabel("Rejected data part, %")
    plt.ylabel("Accuracy, %")
    plt.title(f"Accuracy-Rejection Curves: Static (dashed) vs Adaptive T={plot_temperature} (solid)")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(RESULTS_PLOTS_DIR, "rejection_curves_adaptive.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nRejection curve plot saved to: {plot_path}")

    # --- Save metrics JSON: static baselines + full temperature sweep ---
    import json
    metrics_path = os.path.join(RESULTS_METRICS_DIR, "uncertainty_results_adaptive.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": float(accuracy),
            "n_test": int(n_test),
            "k_neighbors": DEFAULT_K,
            "static_results": results,
            "adaptive_results_by_temperature": adaptive_results_by_temp,
            "suppression_diagnostic_at_T0.5": {
                "corr_weight_concentration_vs_suppression": float(corr_concentration_vs_drop),
                "corr_static_uncertainty_vs_suppression": float(corr_static_unc_vs_drop),
                "mean_static_epistemic": float(static_epistemic.mean()),
                "mean_adaptive_epistemic_T0.5": float(adaptive_epistemic_by_temp[0.5].mean()),
            },
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()