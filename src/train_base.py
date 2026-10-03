"""
train_base.py

[Base Paper, Section 5.2] Trains an ensemble of K=10 MobileNet-V3-small models on hard
ground-truth labels (gt_result). This ensemble's inter-model prediction variance is used
later (in uncertainty.py) as the epistemic uncertainty estimate (Eq. 3 of the base paper).

[Base Paper] Diversity sources across the K models: different train/validation splits,
standard data augmentations (handled in data_loader.py), different random initializations,
and a dropout layer (handled in utils.py's MobileNetV3Reagent).

[Base Paper] Optimizer: Adam (beta1=0.9, beta2=0.999). Loss: cross-entropy (BCEWithLogitsLoss
for our single-logit binary setup). Default 400 epochs, learning rate 4e-4, weight decay 5e-5.

[Extension] The paper states a "scheduled decreasing learning rate" without naming the exact
schedule. We use CosineAnnealingLR over the full epoch count as a reasonable, clearly-flagged
choice.

Usage:
    python src/train_base.py                      # full run: K=10 models, 400 epochs each
    python src/train_base.py --epochs 5 --num_models 2   # quick smoke test
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow running this script directly via `python src/train_base.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from utils import MobileNetV3Reagent, count_parameters

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")

# [Base Paper, Sec 5.2] hyperparameters
DEFAULT_NUM_MODELS = 10
DEFAULT_EPOCHS = 400
DEFAULT_LR = 4e-4
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_BATCH_SIZE = 32
BASE_SPLIT_SEED = 42  # per-model split seed = BASE_SPLIT_SEED + model_index


def train_one_model(
    model_index: int,
    num_reagents: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    use_wandb: bool,
) -> dict:
    """Trains a single ensemble member and saves its checkpoint. Returns a metrics dict."""

    # [Base Paper] "different train/validation splits" per model — achieved via a
    # per-model split_seed, and [Extension] different random init/dropout stream
    # via torch.manual_seed, both varying with model_index.
    torch.manual_seed(model_index)

    train_loader, val_loader, _, reagent_vocab = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        split_seed=BASE_SPLIT_SEED + model_index,
    )
    assert len(reagent_vocab) == num_reagents, (
        f"Reagent vocabulary size mismatch: expected {num_reagents}, got {len(reagent_vocab)}"
    )

    model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=True, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    if use_wandb:
        import wandb
        wandb.init(
            project="medical-ai-base-ensemble",
            name=f"base_model_{model_index}",
            config={
                "model_index": model_index,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "split_seed": BASE_SPLIT_SEED + model_index,
            },
            reinit=True,
        )

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{model_index}.pt")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, reagents, hard_labels, _soft_labels in train_loader:
            images, reagents, hard_labels = images.to(device), reagents.to(device), hard_labels.to(device)

            optimizer.zero_grad()
            logits = model(images, reagents)
            loss = criterion(logits, hard_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == hard_labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, reagents, hard_labels, _soft_labels in val_loader:
                images, reagents, hard_labels = images.to(device), reagents.to(device), hard_labels.to(device)
                logits = model(images, reagents)
                loss = criterion(logits, hard_labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == hard_labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(
                f"  [Model {model_index}] Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # [Base Paper] The paper trains for a fixed epoch count with no stated early-stopping
    # or best-checkpoint selection, so we save the final-epoch weights to stay faithful
    # to the described procedure.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "reagent_vocab": reagent_vocab,
        "model_index": model_index,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "best_val_loss": best_val_loss,
    }, checkpoint_path)

    if use_wandb:
        import wandb
        wandb.finish()

    return {
        "model_index": model_index,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "checkpoint_path": checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the base (hard-label) ensemble.")
    parser.add_argument("--num_models", type=int, default=DEFAULT_NUM_MODELS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training {args.num_models} models for {args.epochs} epochs each.\n")

    # Determine num_reagents once up front from the data loader's vocabulary.
    _, _, _, reagent_vocab = get_dataloaders(batch_size=args.batch_size, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)
    print(f"Reagent vocabulary size: {num_reagents}\n")

    results = []
    overall_start = time.time()

    for i in range(args.num_models):
        print(f"--- Training model {i + 1}/{args.num_models} ---")
        model_start = time.time()

        result = train_one_model(
            model_index=i,
            num_reagents=num_reagents,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            use_wandb=not args.no_wandb,
        )
        results.append(result)

        elapsed = time.time() - model_start
        print(f"  Model {i} done in {elapsed:.1f}s | final_val_acc={result['final_val_acc']:.4f}\n")

    total_elapsed = time.time() - overall_start
    print(f"=== Base ensemble training complete in {total_elapsed / 60:.1f} minutes ===")
    for r in results:
        print(f"  Model {r['model_index']}: final_val_loss={r['final_val_loss']:.4f}, "
              f"final_val_acc={r['final_val_acc']:.4f}, saved to {r['checkpoint_path']}")

    avg_val_acc = sum(r["final_val_acc"] for r in results) / len(results)
    print(f"\nAverage final val accuracy across ensemble: {avg_val_acc:.4f}")


if __name__ == "__main__":
    main()