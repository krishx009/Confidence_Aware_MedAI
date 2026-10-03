"""
build_demo_cache.py

[Extension, Demo] Precomputes everything the Gradio demo needs into a single JSON cache,
so the demo itself requires no PyTorch, no GPU, and no model loading at runtime -- just
this cache file plus the raw test images.

Reuses the exact same validated functions as evaluate_expert_reliability.py (same
ensembles, same reliability weights, same uncertainty decomposition) -- this script adds
no new computation, it only packages existing, already-validated results for display.

Compares two configurations, matching the demo's "Static CAE vs Fully Combined" framing:
    - Static CAE:      base paper's method (uniform model weights, uniform expert labels)
    - Fully Combined:  this project's complete contribution (model-level reliability
                        weighting + expert-reliability-trained CAE ensemble)
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_all_records,
    load_split_key_sets,
    filter_benchmark_records,
    get_dataloaders,
)
from uncertainty import (
    load_ensemble,
    collect_predictions,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    rejection_curve,
    compute_aac,
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
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
CACHE_PATH = os.path.join(DEMO_DIR, "demo_cache.json")

BASE_SPLIT_SEED = 42
ADAPTIVE_TEMPERATURE = 0.05


def auroc_misclassification(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    misclassified = (~correct.astype(bool)).astype(int)
    if misclassified.sum() == 0 or misclassified.sum() == len(misclassified):
        return float("nan")
    return float(roc_auc_score(misclassified, uncertainty))


def percentile_ranks(uncertainty: np.ndarray) -> np.ndarray:
    """
    Returns, for each sample, its percentile rank by uncertainty (0 = most uncertain,
    100 = least uncertain). Used by the demo to answer "would this sample be flagged
    at rejection threshold T%?" via a simple rank < T check, with no recomputation.
    """
    order = np.argsort(-uncertainty)  # descending: most uncertain first
    n = len(uncertainty)
    rank_pct = np.zeros(n)
    for position, idx in enumerate(order):
        rank_pct[idx] = (position / n) * 100.0
    return rank_pct


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs(DEMO_DIR, exist_ok=True)

    _, _, test_loader, reagent_vocab = get_dataloaders(batch_size=32, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)

    print("Loading ensembles...")
    base_models = load_ensemble(BASE_CHECKPOINT_DIR, num_reagents, device)
    cae_static_models = load_ensemble(CAE_CHECKPOINT_DIR, num_reagents, device)
    cae_expert_rel_models = load_ensemble(CAE_EXPERT_REL_CHECKPOINT_DIR, num_reagents, device)

    print("Running inference on test set...")
    base_predictions, hard_labels = collect_predictions(base_models, test_loader, device)
    cae_static_predictions, _ = collect_predictions(cae_static_models, test_loader, device)
    cae_expert_rel_predictions, _ = collect_predictions(cae_expert_rel_models, test_loader, device)

    mean_base_pred = base_predictions.mean(axis=0)
    hard_preds = (mean_base_pred >= 0.5).astype(int)
    correct = (hard_preds == hard_labels.astype(int))

    print("Building model-level reliability weights...")
    base_reference_sets = build_all_reference_sets(base_models, BASE_SPLIT_SEED, batch_size=32, device=device)
    base_test_embeddings = extract_test_embeddings(base_models, test_loader, device)
    weights_base = compute_reliability_weights(
        base_test_embeddings, base_reference_sets, k=DEFAULT_K, temperature=ADAPTIVE_TEMPERATURE
    )

    static_epistemic = epistemic_uncertainty(base_predictions)
    adaptive_epistemic = weighted_epistemic_uncertainty(base_predictions, weights_base)
    aleatoric_static_cae = aleatoric_uncertainty(cae_static_predictions)
    aleatoric_expert_rel_cae = aleatoric_uncertainty(cae_expert_rel_predictions)

    static_uncertainty = static_epistemic + aleatoric_static_cae
    combined_uncertainty = adaptive_epistemic + aleatoric_expert_rel_cae

    print("Computing percentile ranks and rejection curves...")
    static_rank_pct = percentile_ranks(static_uncertainty)
    combined_rank_pct = percentile_ranks(combined_uncertainty)

    thresholds = np.linspace(0.0, 0.99, 100)
    static_curve = rejection_curve(static_uncertainty, correct, thresholds) * 100
    combined_curve = rejection_curve(combined_uncertainty, correct, thresholds) * 100

    static_aac = compute_aac(static_uncertainty, correct, thresholds) * 1e4
    combined_aac = compute_aac(combined_uncertainty, correct, thresholds) * 1e4
    static_auroc = auroc_misclassification(static_uncertainty, correct)
    combined_auroc = auroc_misclassification(combined_uncertainty, correct)

    # --- Recover filenames and reagent for every test-set sample, in loader order ---
    all_records = load_all_records()
    train_pool_keys, test_keys = load_split_key_sets()
    filtered_records = filter_benchmark_records(all_records, train_pool_keys + test_keys)
    # test_loader iterates test_keys in order (shuffle=False, confirmed in data_loader.py)
    ordered_test_keys = test_keys

    assert len(ordered_test_keys) == len(hard_labels), (
        f"Mismatch: {len(ordered_test_keys)} test keys vs {len(hard_labels)} predictions. "
        f"test_loader's iteration order may not match test_keys -- verify before trusting the cache."
    )

    samples = []
    for i, filename in enumerate(ordered_test_keys):
        record = filtered_records[filename]
        samples.append({
            "filename": filename,
            "reagent": record["reagent"],
            "hard_label": int(hard_labels[i]),
            "pred_prob": float(mean_base_pred[i]),
            "correct": bool(correct[i]),
            "static_uncertainty": float(static_uncertainty[i]),
            "combined_uncertainty": float(combined_uncertainty[i]),
            "static_rank_pct": float(static_rank_pct[i]),
            "combined_rank_pct": float(combined_rank_pct[i]),
        })

    cache = {
        "n_test": len(samples),
        "overall_accuracy": float(correct.mean()),
        "images_dir_relative": os.path.join("data", "raw", "imgs", "alvs_dataset_all"),
        "summary": {
            "static": {"aac_x1e4": float(static_aac), "auroc": float(static_auroc)},
            "combined": {"aac_x1e4": float(combined_aac), "auroc": float(combined_auroc)},
        },
        "rejection_curve": {
            "thresholds_pct": (thresholds * 100).tolist(),
            "static_accuracy_pct": static_curve.tolist(),
            "combined_accuracy_pct": combined_curve.tolist(),
        },
        "samples": samples,
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"\nCache written to: {CACHE_PATH}")
    print(f"  {len(samples)} samples, overall accuracy {correct.mean():.4f}")
    print(f"  Static CAE:     AAC={static_aac:.2f}, AUROC={static_auroc:.4f}")
    print(f"  Fully Combined: AAC={combined_aac:.2f}, AUROC={combined_auroc:.4f}")


if __name__ == "__main__":
    main()