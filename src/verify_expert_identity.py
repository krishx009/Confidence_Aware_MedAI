"""
verify_expert_identity.py

[Extension, Phase 2b] Diagnostic check: is the position of an expert within
expert_votes_positions (index 0-5) consistently the same physical expert across the
whole dataset, or could it be re-shuffled per image / per session?

This matters because the planned expert-reliability extension computes one reliability
score per position index (treated as "Expert 0", "Expert 1", ..., "Expert 5"). If position
identity isn't stable, that reliability score would be meaningless -- it would average
together votes from different people under one label.

Method: no explicit expert-ID field exists in the data, so this is checked indirectly.
Images are grouped into batches by filename prefix (which encodes photo session /
collection batch). For each batch, the 6 positions are ranked by their accuracy against
the hard ground-truth label. If position identity is stable across the dataset, the same
position(s) should tend to rank as most/least accurate across most batches (a given
expert's skill level doesn't reset between sessions). If positions were shuffled per
image or per session, this ranking would look close to random from batch to batch.
"""

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_records, load_split_key_sets, filter_benchmark_records

VOTE_BIN_TO_PROB = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}


def extract_batch_id(filename: str) -> str:
    """
    Extracts a session/batch identifier from the filename, based on the two naming
    patterns observed in this dataset:
        "01. CcEeCw_0_0.png"              -> batch "01"
        "2022.04.14_004_map_1_0.png"      -> batch "2022.04.14_004"
    """
    m = re.match(r'^(\d+)\.\s', filename)
    if m:
        return f"session_{m.group(1)}"

    m = re.match(r'^(\d{4}\.\d{2}\.\d{2}_\d+)', filename)
    if m:
        return m.group(1)

    return "unknown"


def main():
    all_records = load_all_records()
    train_pool_keys, test_keys = load_split_key_sets()
    benchmark_keys = train_pool_keys + test_keys
    records = filter_benchmark_records(all_records, benchmark_keys)

    print(f"Total benchmark images: {len(records)}\n")

    # Group images by batch, and for each batch, compute per-position accuracy.
    batch_position_correct = defaultdict(lambda: [[] for _ in range(6)])

    for filename, record in records.items():
        batch = extract_batch_id(filename)
        gt = record["gt_result"]
        positions = record["expert_votes_positions"]
        for i, bin_idx in enumerate(positions):
            prob = VOTE_BIN_TO_PROB[bin_idx]
            expert_hard_vote = 1 if prob >= 0.5 else 0
            batch_position_correct[batch][i].append(1 if expert_hard_vote == gt else 0)

    batches = sorted(batch_position_correct.keys())
    print(f"Number of distinct batches/sessions found: {len(batches)}")
    print(f"Batches: {batches}\n")

    # Compute per-batch, per-position accuracy (only for batches with enough samples).
    MIN_BATCH_SIZE = 20
    batch_accuracies = {}  # batch -> list of 6 accuracies
    for batch in batches:
        sizes = [len(v) for v in batch_position_correct[batch]]
        if min(sizes) < MIN_BATCH_SIZE:
            continue
        accs = [np.mean(v) for v in batch_position_correct[batch]]
        batch_accuracies[batch] = accs

    print(f"Batches with >= {MIN_BATCH_SIZE} samples per position: {len(batch_accuracies)}\n")

    if len(batch_accuracies) < 2:
        print("Not enough batches with sufficient data to test consistency. "
              "Cannot verify expert identity stability this way.")
        return

    # Print per-batch position accuracies and rankings
    print(f"{'Batch':<20} | " + " | ".join(f"Pos{i}" for i in range(6)) + " | Best pos | Worst pos")
    print("-" * 90)
    rankings = []
    for batch, accs in batch_accuracies.items():
        ranking = np.argsort(accs)[::-1]  # best to worst position indices
        rankings.append(ranking)
        acc_str = " | ".join(f"{a:.3f}" for a in accs)
        print(f"{batch:<20} | {acc_str} |    {ranking[0]}     |    {ranking[-1]}")

    # Consistency check 1: how often does each position appear as "best" or "worst"?
    best_counts = defaultdict(int)
    worst_counts = defaultdict(int)
    for ranking in rankings:
        best_counts[ranking[0]] += 1
        worst_counts[ranking[-1]] += 1

    print(f"\nHow often each position ranks BEST across {len(rankings)} batches:")
    for i in range(6):
        print(f"  Position {i}: {best_counts[i]} times ({100*best_counts[i]/len(rankings):.1f}%)")

    print(f"\nHow often each position ranks WORST across {len(rankings)} batches:")
    for i in range(6):
        print(f"  Position {i}: {worst_counts[i]} times ({100*worst_counts[i]/len(rankings):.1f}%)")

    # Consistency check 2: average Spearman rank correlation between all pairs of batches
    from scipy.stats import spearmanr
    correlations = []
    accs_list = list(batch_accuracies.values())
    for i in range(len(accs_list)):
        for j in range(i + 1, len(accs_list)):
            corr, _ = spearmanr(accs_list[i], accs_list[j])
            correlations.append(corr)

    mean_corr = np.mean(correlations)
    print(f"\nMean pairwise Spearman rank correlation between batches' position-accuracy "
          f"rankings: {mean_corr:.3f}")
    print("(1.0 = identical ranking every time -> strong evidence of stable expert identity;")
    print(" 0.0 = no relationship -> positions are likely shuffled/anonymized per batch)")

    # Overall per-position accuracy across the full dataset, for reference
    print(f"\nOverall per-position accuracy across all {len(records)} benchmark images:")
    all_correct = [[] for _ in range(6)]
    for filename, record in records.items():
        gt = record["gt_result"]
        for i, bin_idx in enumerate(record["expert_votes_positions"]):
            prob = VOTE_BIN_TO_PROB[bin_idx]
            vote = 1 if prob >= 0.5 else 0
            all_correct[i].append(1 if vote == gt else 0)
    for i in range(6):
        print(f"  Position {i}: accuracy = {np.mean(all_correct[i]):.4f}, n = {len(all_correct[i])}")


if __name__ == "__main__":
    main()