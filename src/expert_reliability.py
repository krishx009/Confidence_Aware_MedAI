"""
expert_reliability.py

[Extension, Phase 2b] Computes reliability-weighted soft labels using per-expert
reliability scores, extending the base paper's uniform mean-of-6-experts soft label
(y~ = (1/6) * sum of 6 expert values) to a reliability-weighted average instead.

Verified precondition (see verify_expert_identity.py): expert position index (0-5) is
stable across the entire dataset -- the same physical expert consistently occupies a
given position across 58 independent collection batches spanning 2019-2022, confirmed by
one position ranking "best" in 87.9% of batches and another ranking "worst" in 56.9%.

Per-expert reliability combines two components, per the project's abstract ("consistency
and historical performance"):

    1. Historical performance [adaptive, k-NN]: for image x, find x's k nearest neighbors
       (by averaged 10-model embedding) within the 623-image train pool (excluding x
       itself to prevent trivial self-match leakage), then average that expert's
       continuous "closeness" score (1 - |vote_prob - gt_result|) over those neighbors.
       Continuous closeness is used rather than binary correctness from the outset,
       reusing the lesson learned in Phase 2's model-level reliability estimator: expert
       accuracies here are already 90-99.6%, so a binary signal would saturate the same
       way the model-level one did before that fix.

    2. Consistency [direct, no lookup needed]: for image x, how much does this expert's
       vote agree with the mean of the OTHER 5 experts' votes on that same image --
       1 - |vote_e(x) - mean(other 5 votes on x)|. This requires no historical reference
       set at all, since peer votes on the same image are already available for every
       image, train or test.

The two components are averaged (equal weight, a simple and defensible default absent a
prescribed blend ratio), then combined into per-sample weights across the 6 experts using
the same z-score + temperature-softmax recipe validated for model-level reliability in
Phase 2 (chosen deliberately, since raw-value normalization was already shown to collapse
toward uniform when candidate scores share a similarly high absolute magnitude).
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_all_records,
    load_split_key_sets,
    filter_benchmark_records,
    build_reagent_vocabulary,
    BloodyWellDataset,
)
from utils import MobileNetV3Reagent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
NUM_MODELS = 10
NUM_EXPERTS = 6

VOTE_BIN_TO_PROB = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}

DEFAULT_K = 15
DEFAULT_TEMPERATURE = 0.5


# ---------------------------------------------------------------------------
# Averaged base-ensemble embeddings for the train pool
# ---------------------------------------------------------------------------

def load_base_models(num_reagents: int, device: torch.device):
    models = []
    for i in range(NUM_MODELS):
        checkpoint = torch.load(os.path.join(BASE_CHECKPOINT_DIR, f"model_{i}.pt"), map_location=device)
        model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def compute_averaged_embeddings(models, keys, filtered_records, reagent_vocab, device: torch.device):
    """
    Returns (embeddings, key_order) where embeddings[i] is the average of all 10 base
    models' embeddings for keys[i]. No augmentation is applied (deterministic features).
    """
    dataset = BloodyWellDataset(keys, filtered_records, reagent_vocab, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_embeddings = []
    with torch.no_grad():
        for images, reagents, _hard_labels, _soft_labels in loader:
            images, reagents = images.to(device), reagents.to(device)
            batch_embeddings = torch.zeros(images.size(0), models[0].fc1.in_features, device=device)
            for model in models:
                batch_embeddings += model.get_embedding(images, reagents)
            batch_embeddings /= len(models)
            all_embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, keys  # keys preserved in the same order as the (non-shuffled) loader


# ---------------------------------------------------------------------------
# Per-expert component scores
# ---------------------------------------------------------------------------

def compute_expert_probs_and_closeness(keys, filtered_records):
    """
    Returns:
        expert_probs: (N, 6) -- each expert's vote converted to a probability
        closeness: (N, 6) -- 1 - |prob - gt_result| per expert, per image
        gt: (N,) -- hard ground-truth labels
    """
    n = len(keys)
    expert_probs = np.zeros((n, NUM_EXPERTS))
    closeness = np.zeros((n, NUM_EXPERTS))
    gt = np.zeros(n)

    for idx, key in enumerate(keys):
        record = filtered_records[key]
        gt[idx] = record["gt_result"]
        positions = record["expert_votes_positions"]
        for e in range(NUM_EXPERTS):
            prob = VOTE_BIN_TO_PROB[positions[e]]
            expert_probs[idx, e] = prob
            closeness[idx, e] = 1.0 - abs(prob - gt[idx])

    return expert_probs, closeness, gt


def compute_historical_performance(embeddings, closeness, k: int = DEFAULT_K):
    """
    [Adaptive] For each image and each expert, average that expert's closeness score
    over the image's k nearest neighbors in embedding space, EXCLUDING the image itself
    (to prevent trivial self-match leakage within the train pool).

    Returns: historical_performance, shape (N, 6).
    """
    n = embeddings.shape[0]
    k_query = min(k + 1, n)  # +1 so we can drop the self-match

    nn_finder = NearestNeighbors(n_neighbors=k_query, metric="cosine")
    nn_finder.fit(embeddings)
    _, neighbor_indices = nn_finder.kneighbors(embeddings)  # (N, k_query), includes self at some position

    historical_performance = np.zeros((n, NUM_EXPERTS))
    for i in range(n):
        neighbors = [idx for idx in neighbor_indices[i] if idx != i][:k]
        historical_performance[i] = closeness[neighbors].mean(axis=0)

    return historical_performance


def compute_consistency(expert_probs):
    """
    [Direct, no lookup] For each image and each expert, 1 - |own vote - mean of the
    other 5 experts' votes on the same image|. Requires no historical reference set.

    Returns: consistency, shape (N, 6).
    """
    n = expert_probs.shape[0]
    consistency = np.zeros((n, NUM_EXPERTS))
    for e in range(NUM_EXPERTS):
        other_experts = [i for i in range(NUM_EXPERTS) if i != e]
        peer_mean = expert_probs[:, other_experts].mean(axis=1)
        consistency[:, e] = 1.0 - np.abs(expert_probs[:, e] - peer_mean)
    return consistency


# ---------------------------------------------------------------------------
# Combine into weights and final reliability-weighted soft label
# ---------------------------------------------------------------------------

def compute_expert_weights(historical_performance, consistency, temperature: float = DEFAULT_TEMPERATURE):
    """
    Combines the two components (equal-weighted average), then z-scores across the 6
    experts per image and applies a temperature-scaled softmax -- the same recipe
    validated for model-level reliability weighting in Phase 2.

    Returns: weights, shape (N, 6), each row summing to 1.
    """
    raw_reliability = 0.5 * historical_performance + 0.5 * consistency  # (N, 6)

    per_sample_mean = raw_reliability.mean(axis=1, keepdims=True)
    per_sample_std = raw_reliability.std(axis=1, keepdims=True)
    z_scores = (raw_reliability - per_sample_mean) / (per_sample_std + 1e-8)

    scaled = z_scores / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    return weights


def compute_reliability_weighted_soft_labels(expert_probs, weights):
    """[Extension] y~(x) = sum_e w_e(x) * prob_e(x), replacing the uniform mean-of-6."""
    return np.sum(weights * expert_probs, axis=1)


# ---------------------------------------------------------------------------
# Self-test / generation script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    all_records = load_all_records()
    train_pool_keys, test_keys = load_split_key_sets()
    benchmark_keys = train_pool_keys + test_keys
    filtered_records = filter_benchmark_records(all_records, benchmark_keys)
    reagent_vocab = build_reagent_vocabulary(filtered_records)
    num_reagents = len(reagent_vocab)

    print(f"Train pool size (reference set + target images): {len(train_pool_keys)}\n")

    print("Loading base ensemble...")
    base_models = load_base_models(num_reagents, device)

    print("Computing averaged 10-model embeddings for the train pool (no augmentation)...")
    embeddings, ordered_keys = compute_averaged_embeddings(
        base_models, train_pool_keys, filtered_records, reagent_vocab, device
    )
    print(f"  Embeddings shape: {embeddings.shape}\n")

    print("Computing per-expert probabilities, closeness, and ground truth...")
    expert_probs, closeness, gt = compute_expert_probs_and_closeness(ordered_keys, filtered_records)

    print("Computing historical performance (adaptive k-NN, self excluded)...")
    historical_performance = compute_historical_performance(embeddings, closeness, k=DEFAULT_K)

    print("Computing consistency (direct, peer votes on same image)...")
    consistency = compute_consistency(expert_probs)

    print(f"Combining into reliability weights (temperature={DEFAULT_TEMPERATURE})...\n")
    weights = compute_expert_weights(historical_performance, consistency, temperature=DEFAULT_TEMPERATURE)

    # --- Diagnostics ---
    uniform_weight = 1.0 / NUM_EXPERTS
    deviation = np.abs(weights - uniform_weight).mean()
    print("=" * 70)
    print("Diagnostics")
    print("-" * 70)
    print(f"Mean absolute deviation from uniform (1/6={uniform_weight:.4f}): {deviation:.4f}")
    print(f"\nAverage weight given to each expert position (across all {len(ordered_keys)} images):")
    for e in range(NUM_EXPERTS):
        print(f"  Position {e}: avg weight = {weights[:, e].mean():.4f}  "
              f"(overall accuracy this position: {1 - np.abs(expert_probs[:, e] - gt).round().mean():.4f})")

    print(f"\nSanity check -- Position 5 (previously shown most accurate) vs "
          f"Position 1 (previously shown least accurate):")
    print(f"  Position 5 avg weight: {weights[:, 5].mean():.4f}")
    print(f"  Position 1 avg weight: {weights[:, 1].mean():.4f}")
    print("  (Position 5's weight should be clearly higher than Position 1's if the "
          "mechanism is behaving sensibly.)")

    print(f"\nComponent breakdown per position (to explain any ranking reversals vs raw accuracy):")
    print(f"{'Position':<10} | {'Accuracy':>9} | {'Hist.Perf':>10} | {'Consistency':>12} | {'Combined(raw)':>14} | {'Final Weight':>12}")
    raw_combined = 0.5 * historical_performance + 0.5 * consistency
    for e in range(NUM_EXPERTS):
        acc = 1 - np.abs(expert_probs[:, e] - gt).round().mean()
        print(f"{e:<10} | {acc:>9.4f} | {historical_performance[:, e].mean():>10.4f} | "
              f"{consistency[:, e].mean():>12.4f} | {raw_combined[:, e].mean():>14.4f} | {weights[:, e].mean():>12.4f}")
    print("=" * 70)

    # --- Compute and save the new reliability-weighted soft labels ---
    new_soft_labels = compute_reliability_weighted_soft_labels(expert_probs, weights)
    old_soft_labels = expert_probs.mean(axis=1)  # the original uniform-mean soft label

    label_diff = np.abs(new_soft_labels - old_soft_labels)
    print(f"\nMean absolute difference between new and old soft labels: {label_diff.mean():.4f}")
    print(f"Max absolute difference: {label_diff.max():.4f}")

    output = {
        key: float(new_soft_labels[i]) for i, key in enumerate(ordered_keys)
    }
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "reliability_weighted_soft_labels.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nReliability-weighted soft labels saved to: {output_path}")