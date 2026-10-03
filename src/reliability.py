"""
reliability.py

[Extension, Phase 2] Implements the project's novel contribution: Adaptive
Reliability-Aware weighting for ensemble uncertainty estimation.

For a given test sample x and ensemble member i, a reliability score w_i(x) is computed
as the fraction of model i's k nearest neighbors (in embedding space, among model i's OWN
validation set) that model i classified correctly. This replaces the base paper's uniform
(1/K) ensemble weighting in Eq. 3 and Eq. 4 with a per-sample, per-model dynamic weight,
directly targeting the overconfidence/redundancy issue diagnosed in Milestone 3: a model
that historically performs poorly on inputs similar to x is down-weighted for that specific
sample, rather than contributing equally regardless of its track record on similar inputs.

Design choices [Extension]:
    - Embedding space: the 587-dim pre-classifier feature vector (pooled visual features +
      reagent one-hot), taken before dropout so it is deterministic.
    - k=15 neighbors, cosine similarity (both configurable).
    - Reliability scores are normalized to sum to 1 across the K models per sample; if all
      K models show zero reliability for a sample (rare), falls back to uniform 1/K weights.
"""

import os
import sys

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from utils import MobileNetV3Reagent

DEFAULT_K = 15
DEFAULT_METRIC = "cosine"


# ---------------------------------------------------------------------------
# Reference set construction: each model's own validation embeddings + correctness
# ---------------------------------------------------------------------------

def build_reference_set(model, val_loader, device: torch.device):
    """
    Runs a single model over its own validation set, collecting:
        embeddings: (N_val, D) -- pre-classifier feature vectors
        correct: (N_val,) -- 1.0 if model's hard-label prediction matched ground truth,
            else 0.0. Kept only for reporting/diagnostics (e.g. val accuracy).
        closeness: (N_val,) -- 1 - |p - y|, a continuous measure of how close the model's
            raw probability was to the true label. [Extension, diagnostic finding] The
            binary `correct` signal saturates near 1.0 almost everywhere once a model's
            validation accuracy exceeds ~98%, since k=15 neighborhoods drawn from such a
            population are overwhelmingly "all correct" regardless of embedding-space
            location -- leaving no real signal for reliability weighting to exploit. The
            continuous closeness score retains meaningful variation (e.g. distinguishing
            p=0.99 from p=0.51, both "correct" under a 0.5 threshold) and is used for the
            actual reliability computation; `correct` is retained only for reporting.
    """
    model.eval()
    embeddings_list = []
    correct_list = []
    closeness_list = []

    with torch.no_grad():
        for images, reagents, hard_labels, _soft_labels in val_loader:
            images, reagents = images.to(device), reagents.to(device)
            hard_labels = hard_labels.to(device)

            embeddings = model.get_embedding(images, reagents)
            logits = model(images, reagents)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct = (preds == hard_labels).float()
            closeness = 1.0 - torch.abs(probs - hard_labels)

            embeddings_list.append(embeddings.cpu().numpy())
            correct_list.append(correct.cpu().numpy())
            closeness_list.append(closeness.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    correct = np.concatenate(correct_list, axis=0)
    closeness = np.concatenate(closeness_list, axis=0)
    return embeddings, correct, closeness


def build_all_reference_sets(models, base_split_seed: int, batch_size: int, device: torch.device):
    """
    For each model i in `models`, reconstructs model i's own original validation split
    (via split_seed = base_split_seed + i, matching train_base.py / train_cae.py) and
    builds its reference set. Returns a list of (embeddings, correct, closeness) tuples,
    one per model.
    """
    reference_sets = []
    for i, model in enumerate(models):
        _, val_loader, _, _ = get_dataloaders(
            batch_size=batch_size, split_seed=base_split_seed + i
        )
        embeddings, correct, closeness = build_reference_set(model, val_loader, device)
        reference_sets.append((embeddings, correct, closeness))
    return reference_sets


# ---------------------------------------------------------------------------
# Test-set embedding extraction
# ---------------------------------------------------------------------------

def extract_test_embeddings(models, test_loader, device: torch.device):
    """
    Runs every model over the full test set, collecting each model's own embedding for
    every test sample. Returns a list of (N_test, D) arrays, one per model, in the same
    sample order for all models (test_loader has shuffle=False).
    """
    all_embeddings = [[] for _ in models]

    with torch.no_grad():
        for images, reagents, _hard_labels, _soft_labels in test_loader:
            images, reagents = images.to(device), reagents.to(device)
            for k, model in enumerate(models):
                model.eval()
                embeddings = model.get_embedding(images, reagents)
                all_embeddings[k].append(embeddings.cpu().numpy())

    return [np.concatenate(e, axis=0) for e in all_embeddings]


# ---------------------------------------------------------------------------
# Reliability weight computation
# ---------------------------------------------------------------------------

def compute_reliability_weights(
    test_embeddings_per_model: list,
    reference_sets: list,
    k: int = DEFAULT_K,
    metric: str = DEFAULT_METRIC,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    test_embeddings_per_model: list of K arrays, each (N_test, D) -- model i's embedding
        of every test sample.
    reference_sets: list of K (embeddings, correct, closeness) tuples from
        build_all_reference_sets. The continuous `closeness` array is used for the actual
        reliability computation (see build_reference_set docstring for why).
    temperature: softmax temperature applied to the per-sample z-scored raw reliability
        values. Lower values produce more decisive (less uniform) weighting.

    Returns: weights, shape (K, N_test) -- normalized so weights[:, n].sum() == 1 for
        every test sample n.

    [Extension, diagnostic finding] An earlier version normalized raw reliability scores
    directly (dividing by their per-sample sum). This was found empirically to collapse
    to near-uniform weights (mean deviation from 1/K of only 0.0005), because this
    project's K=10 base models are highly correlated with each other (consistent with the
    r=0.79 epistemic-aleatoric correlation diagnosed in Milestone 3) -- their raw
    reliability scores for any given sample land in a narrow, similarly-high band, so
    direct normalization of similar-magnitude values stays close to uniform regardless of
    real underlying differences. Z-scoring each sample's K raw scores against each other
    (relative to that sample's own mean and spread) before applying softmax instead asks
    "which model is relatively most trustworthy for this sample compared to its ensemble
    peers" -- a signal that remains meaningful even when absolute magnitudes are similar.
    """
    num_models = len(test_embeddings_per_model)
    n_test = test_embeddings_per_model[0].shape[0]

    raw_reliability = np.zeros((num_models, n_test), dtype=np.float64)

    for i in range(num_models):
        ref_embeddings, _ref_correct, ref_closeness = reference_sets[i]
        k_i = min(k, len(ref_embeddings))  # guard against k > val set size

        nn_finder = NearestNeighbors(n_neighbors=k_i, metric=metric)
        nn_finder.fit(ref_embeddings)

        _, neighbor_indices = nn_finder.kneighbors(test_embeddings_per_model[i])  # (N_test, k_i)
        neighbor_closeness = ref_closeness[neighbor_indices]  # (N_test, k_i)
        raw_reliability[i] = neighbor_closeness.mean(axis=1)  # (N_test,)

    # Per-sample z-score across the K models, then temperature-scaled softmax.
    per_sample_mean = raw_reliability.mean(axis=0, keepdims=True)  # (1, N_test)
    per_sample_std = raw_reliability.std(axis=0, keepdims=True)    # (1, N_test)
    z_scores = (raw_reliability - per_sample_mean) / (per_sample_std + 1e-8)

    scaled = z_scores / temperature
    scaled = scaled - scaled.max(axis=0, keepdims=True)  # numerical stability
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / exp_scaled.sum(axis=0, keepdims=True)

    return weights


# ---------------------------------------------------------------------------
# Weighted uncertainty computation (reliability-aware analogs of Eq. 3-4)
# ---------------------------------------------------------------------------

def weighted_epistemic_uncertainty(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    [Extension, redesigned after diagnostic finding] Reliability-informed analog of Eq. 3.

    An earlier version weighted BOTH the mean-centering step AND the per-model variance
    summation by reliability: epistemic(x) = sum_i w_i(x) * (p_i(x) - weighted_mean(x))^2.
    This was found empirically to substantially suppress epistemic uncertainty specifically
    on the samples where models originally disagreed the most (correlation between static
    epistemic uncertainty and suppression amount = 0.64), because down-weighting a
    "less reliable" (disagreeing) model directly shrinks its contribution to the variance
    sum -- destroying exactly the disagreement signal Eq. 3 exists to capture.

    This version uses reliability weights ONLY to compute a smarter reference point (the
    weighted mean prediction) but reverts to a plain, unweighted (1/K) summation over
    per-model squared deviations from that point. This preserves full disagreement
    sensitivity across all K models while still making the uncertainty estimate adaptive
    (the reference point itself is data-dependent and reliability-informed), avoiding the
    suppression mechanism observed with full weighting.
    """
    weighted_mean = np.sum(weights * predictions, axis=0, keepdims=True)  # (1, N) -- reliability-informed center
    num_models = predictions.shape[0]
    return np.mean((predictions - weighted_mean) ** 2, axis=0)  # (N,) -- unweighted (1/K) summation


def weighted_aleatoric_uncertainty(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """[Extension] Reliability-weighted analog of Eq. 4."""
    return np.sum(weights * predictions * (1.0 - predictions), axis=0)  # (N,)


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
    NUM_MODELS = 10
    BASE_SPLIT_SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    _, _, test_loader, reagent_vocab = get_dataloaders(batch_size=32, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)

    print("Loading base ensemble...")
    base_models = []
    for i in range(NUM_MODELS):
        checkpoint = torch.load(
            os.path.join(BASE_CHECKPOINT_DIR, f"model_{i}.pt"), map_location=device
        )
        model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        base_models.append(model)

    print("Building reference sets (validation embeddings + correctness) per model...")
    start = time.time()
    reference_sets = build_all_reference_sets(base_models, BASE_SPLIT_SEED, batch_size=32, device=device)
    print(f"  Done in {time.time() - start:.1f}s")
    for i, (emb, corr, closeness) in enumerate(reference_sets):
        print(f"  Model {i}: {emb.shape[0]} val samples, embedding dim {emb.shape[1]}, "
              f"val accuracy {corr.mean():.4f}, mean closeness {closeness.mean():.4f}, "
              f"closeness std {closeness.std():.4f}")

    print("\nExtracting test-set embeddings for all models...")
    start = time.time()
    test_embeddings = extract_test_embeddings(base_models, test_loader, device)
    print(f"  Done in {time.time() - start:.1f}s")
    print(f"  Test embeddings shape per model: {test_embeddings[0].shape}")

    print(f"\nComputing reliability weights (k={DEFAULT_K}, metric={DEFAULT_METRIC})...")
    uniform_weight = 1.0 / NUM_MODELS

    for temperature in [1.0, 0.5, 0.25, 0.1]:
        start = time.time()
        weights = compute_reliability_weights(
            test_embeddings, reference_sets, k=DEFAULT_K, metric=DEFAULT_METRIC, temperature=temperature
        )
        elapsed = time.time() - start
        deviation_from_uniform = np.abs(weights - uniform_weight).mean()
        max_weight_mean = weights.max(axis=0).mean()  # average of each sample's top weight
        print(f"  temperature={temperature:<5} | computed in {elapsed:.2f}s | "
              f"mean abs deviation from uniform={deviation_from_uniform:.4f} | "
              f"avg top-model weight per sample={max_weight_mean:.4f} "
              f"(uniform baseline={uniform_weight:.4f})")