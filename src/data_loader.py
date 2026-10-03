"""
data_loader.py

Data loading pipeline for the BloodyWell dataset, as used to reproduce and extend
Korchagin et al., "Improving Uncertainty Estimation with Confidence-Aware Training Data"
(WACV 2025), for the project "Adaptive Reliability-Aware Confidence Learning for
Improved Uncertainty Estimation in Medical AI".

[Base Paper] Dataset: BloodyWell, 3,139 labeled blood-typing images, hard labels (gt_result)
    + 6 expert soft labels (expert_votes_positions) per image.
[Base Paper] Backbone: MobileNet-V3-small, with one-hot reagent encoding fed into a dense layer.
[Extension]  The released dataset has no 'val' split_type. We reconstruct one by stratified
    50/50 splitting of the released 'train' pool (train_test_split, stratify=gt_result,
    fixed random_state), recovering the paper's stated ~1:1:8 ratio.
[Extension]  Reagent codes are normalized to lowercase before one-hot encoding, since the raw
    data contains inconsistent casing (e.g. 'C' and 'c') that would otherwise be treated as
    distinct reagent classes.
[Extension]  experts_info.json's own internal 'split_type' field was found to be stale and
    inconsistent with the two dedicated split files (train_dataset_10_10_80.json,
    test_dataset_10_10_80.json) for a large number of images. The dedicated split files are
    therefore treated as the sole source of truth for train/test membership (623 + 2,516 =
    3,139, matching the paper exactly); experts_info.json's own 'split_type' field is ignored
    entirely and used only as a source of per-image label content (gt_result, reagent,
    expert_votes_positions).
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS_INFO_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "experts_info.json")
TRAIN_SPLIT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train_dataset_10_10_80.json")
TEST_SPLIT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "test_dataset_10_10_80.json")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "imgs", "alvs_dataset_all")

IMAGE_SIZE = 224
RANDOM_SEED = 42

# [Base Paper] Expert confidence bins, indices 0-4 map to these probability values.
VOTE_BIN_TO_PROB = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}

# ImageNet normalization stats, standard for MobileNet-V3-small pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Loading and filtering raw records
# ---------------------------------------------------------------------------

def load_all_records() -> Dict[str, dict]:
    """Load experts_info.json and return the raw dict keyed by filename."""
    with open(EXPERTS_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_split_key_sets() -> Tuple[List[str], List[str]]:
    """
    Load the two dedicated split files and return (train_pool_keys, test_keys).

    [Extension] These files are treated as the sole source of truth for train/test
    membership, since experts_info.json's own internal 'split_type' field was found
    to disagree with these files for a large number of images (see module docstring).
    """
    with open(TRAIN_SPLIT_PATH, "r", encoding="utf-8") as f:
        train_split = json.load(f)
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        test_split = json.load(f)

    return list(train_split.keys()), list(test_split.keys())


def filter_benchmark_records(
    all_records: Dict[str, dict], benchmark_keys: List[str]
) -> Dict[str, dict]:
    """
    Keep only records whose filename appears in the authoritative benchmark key set
    (the union of train_dataset_10_10_80.json and test_dataset_10_10_80.json keys).
    This excludes:
      - 484 empty-well images (is_empty=True, never in either split file)
      - 259 valid but unassigned images (not in either split file)
    and leaves exactly the paper's reported 3,139 benchmark images.
    """
    benchmark_key_set = set(benchmark_keys)
    filtered = {k: v for k, v in all_records.items() if k in benchmark_key_set}
    return filtered


def build_reagent_vocabulary(records: Dict[str, dict]) -> List[str]:
    """
    [Extension] Build a sorted, lowercase-normalized vocabulary of reagent codes.
    Raw data contains inconsistent casing (e.g. 'C' vs 'c') that would otherwise be
    treated as separate reagent classes; we normalize before building the vocabulary.
    """
    reagents = set()
    for v in records.values():
        reagents.add(v["reagent"].strip().lower())
    return sorted(reagents)


# ---------------------------------------------------------------------------
# Split reconstruction
# ---------------------------------------------------------------------------

def build_splits(
    filtered_records: Dict[str, dict],
    train_pool_keys: List[str],
    test_keys: List[str],
    split_seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_keys, val_keys, test_keys).

    [Base Paper] test_keys (2,516 images), from test_dataset_10_10_80.json, used as-is.
    [Extension]  train_pool_keys (623 images), from train_dataset_10_10_80.json, are
        stratified 50/50 by gt_result into actual train/val sets. split_seed controls
        this partition — [Base Paper, Sec 5.2] "Each model employs different
        train/validation splits" as one source of ensemble diversity, so each of the
        K ensemble members should call this with a different split_seed.
    """
    train_pool_labels = [filtered_records[k]["gt_result"] for k in train_pool_keys]

    train_keys, val_keys = train_test_split(
        train_pool_keys,
        test_size=0.5,
        stratify=train_pool_labels,
        random_state=split_seed,
    )

    return train_keys, val_keys, test_keys


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def get_hard_label(record: dict) -> float:
    """[Base Paper] Hard ground-truth label, 0 or 1."""
    return float(record["gt_result"])


def get_soft_label(record: dict) -> float:
    """
    [Base Paper] Soft label y~ = (1/6) * sum of 6 expert confidence values.
    expert_votes_positions holds 6 integers in [0, 4], each mapped to a probability
    via VOTE_BIN_TO_PROB, then averaged.
    """
    positions = record["expert_votes_positions"]
    probs = [VOTE_BIN_TO_PROB[p] for p in positions]
    return float(np.mean(probs))


def get_reagent_onehot(record: dict, vocabulary: List[str]) -> torch.Tensor:
    """[Base Paper + Extension] One-hot encode the (case-normalized) reagent code."""
    reagent = record["reagent"].strip().lower()
    onehot = torch.zeros(len(vocabulary), dtype=torch.float32)
    idx = vocabulary.index(reagent)
    onehot[idx] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BloodyWellDataset(Dataset):
    """
    Returns (image_tensor, reagent_onehot, hard_label, soft_label) per sample.

    image_tensor: torch.FloatTensor, shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized.
    reagent_onehot: torch.FloatTensor, shape (num_reagents,).
    hard_label: torch.FloatTensor, scalar, 0.0 or 1.0.
    soft_label: torch.FloatTensor, scalar, in [0.0, 1.0].
    """

    def __init__(
        self,
        keys: List[str],
        records: Dict[str, dict],
        reagent_vocabulary: List[str],
        augment: bool = False,
        reliability_soft_labels: dict = None,
    ):
        self.keys = keys
        self.records = records
        self.reagent_vocabulary = reagent_vocabulary
        # [Extension, Phase 2b] Optional override: when provided, filenames present in
        # this dict use the reliability-weighted soft label (from expert_reliability.py)
        # instead of the base paper's uniform mean-of-6-experts soft label. Filenames not
        # present in the dict (e.g. test-set images, which are never covered by this
        # file) fall back to the original get_soft_label() computation automatically.
        # Defaults to None, which preserves the original behavior exactly.
        self.reliability_soft_labels = reliability_soft_labels or {}

        if augment:
            # [Base Paper] "standard data augmentations" used during base ensemble training
            # to encourage diversity across ensemble members.
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        filename = self.keys[idx]
        record = self.records[filename]

        image_path = os.path.join(IMAGES_DIR, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        reagent_onehot = get_reagent_onehot(record, self.reagent_vocabulary)
        hard_label = torch.tensor(get_hard_label(record), dtype=torch.float32)

        if filename in self.reliability_soft_labels:
            soft_label_value = self.reliability_soft_labels[filename]
        else:
            soft_label_value = get_soft_label(record)
        soft_label = torch.tensor(soft_label_value, dtype=torch.float32)

        return image_tensor, reagent_onehot, hard_label, soft_label


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def load_reliability_weighted_soft_labels() -> Dict[str, float]:
    """
    [Extension, Phase 2b] Loads the per-image reliability-weighted soft labels produced
    by expert_reliability.py. Covers only the 623-image train pool (the only images ever
    used with soft-label supervision); returns an empty dict if the file doesn't exist
    yet, so callers can safely fall back to the original uniform-mean soft label.
    """
    path = os.path.join(PROJECT_ROOT, "data", "processed", "reliability_weighted_soft_labels.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    split_seed: int = RANDOM_SEED,
    use_reliability_weighted_soft_labels: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds and returns (train_loader, val_loader, test_loader, reagent_vocabulary).

    train_loader uses augmentation; val_loader and test_loader do not.
    split_seed controls the train/val partition of the 623-image train pool — pass a
    different value per ensemble member to satisfy the base paper's requirement that
    each model uses a different train/validation split (Sec 5.2).

    [Extension, Phase 2b] use_reliability_weighted_soft_labels: when True, train_loader
    and val_loader use the reliability-weighted soft labels (expert_reliability.py)
    instead of the base paper's uniform mean-of-6-experts soft label. test_loader is
    never affected, since the reliability-weighted labels only cover the 623-image train
    pool and the test set never uses soft labels regardless. Defaults to False, which
    preserves the exact original behavior used by all existing scripts.
    """
    all_records = load_all_records()
    train_pool_keys, test_keys = load_split_key_sets()
    benchmark_keys = train_pool_keys + test_keys

    assert len(set(benchmark_keys)) == 3139, (
        f"Expected 3,139 unique benchmark images from the split files, got "
        f"{len(set(benchmark_keys))}. Check train_dataset_10_10_80.json and "
        f"test_dataset_10_10_80.json for duplicate or unexpected keys."
    )

    filtered_records = filter_benchmark_records(all_records, benchmark_keys)

    assert len(filtered_records) == 3139, (
        f"Expected 3,139 benchmark images after joining with experts_info.json, got "
        f"{len(filtered_records)}. Some split-file keys may be missing from "
        f"experts_info.json — check for a dataset version mismatch."
    )

    reagent_vocabulary = build_reagent_vocabulary(filtered_records)

    train_keys, val_keys, test_keys = build_splits(
        filtered_records, train_pool_keys, test_keys, split_seed=split_seed
    )

    reliability_soft_labels = None
    if use_reliability_weighted_soft_labels:
        reliability_soft_labels = load_reliability_weighted_soft_labels()
        assert len(reliability_soft_labels) > 0, (
            "use_reliability_weighted_soft_labels=True but "
            "data/processed/reliability_weighted_soft_labels.json was not found or is "
            "empty. Run expert_reliability.py first to generate it."
        )

    train_dataset = BloodyWellDataset(
        train_keys, filtered_records, reagent_vocabulary, augment=True,
        reliability_soft_labels=reliability_soft_labels,
    )
    val_dataset = BloodyWellDataset(
        val_keys, filtered_records, reagent_vocabulary, augment=False,
        reliability_soft_labels=reliability_soft_labels,
    )
    test_dataset = BloodyWellDataset(test_keys, filtered_records, reagent_vocabulary, augment=False)

    loader_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        # [Extension — performance] On Windows, DataLoader workers are respawned every
        # epoch by default (spawn start method), which was empirically measured to add
        # ~4x overhead per epoch on this dataset. persistent_workers=True keeps worker
        # processes alive across epochs, paying the spawn cost only once.
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, reagent_vocabulary


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, reagent_vocab = get_dataloaders(batch_size=8)

    print(f"Reagent vocabulary ({len(reagent_vocab)} classes): {reagent_vocab}")
    print(f"Train batches: {len(train_loader)} | samples: {len(train_loader.dataset)}")
    print(f"Val batches:   {len(val_loader)} | samples: {len(val_loader.dataset)}")
    print(f"Test batches:  {len(test_loader)} | samples: {len(test_loader.dataset)}")

    images, reagents, hard_labels, soft_labels = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  images:      {images.shape}")
    print(f"  reagents:    {reagents.shape}")
    print(f"  hard_labels: {hard_labels.shape}, values: {hard_labels.tolist()}")
    print(f"  soft_labels: {soft_labels.shape}, values: {[round(v, 3) for v in soft_labels.tolist()]}")