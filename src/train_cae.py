"""
train_cae.py

[Base Paper, Section 3.3 / Figure 2] Fine-tunes each of the K=10 base ensemble models
(trained on hard labels in train_base.py) on soft expert-confidence labels, producing the
Confidence-Aware Ensemble (CAE). This ensemble's inter-model prediction variance is used
later (in uncertainty.py) as the aleatoric uncertainty estimate (Eq. 4 of the base paper).

[Base Paper] Each CAE model starts from its corresponding trained base-ensemble checkpoint
(not randomly initialized) and is fine-tuned for 300 epochs to predict the mean expert vote
y~ = (1/6) * sum of 6 expert confidence values, using the same train/validation split that
model originally used during base training (reproduced deterministically via the same
split_seed).

[Extension] The base paper states the learning rate (4e-4) and weight decay (5e-5) used for
the initial 400-epoch hard-label training stage, but does not explicitly restate separate
values for the 300-epoch soft-label fine-tuning stage. We reuse the same values with a
fresh cosine-annealing schedule over the 300 fine-tuning epochs, since the paper is silent
on this detail.

Usage:
    python src/train_cae.py                          # full run: K=10 models, 300 epochs each
    python src/train_cae.py --epochs 5 --num_models 2 # quick smoke test
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from utils import MobileNetV3Reagent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
CAE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "cae_ensemble")

# [Base Paper, Sec 5.2] fine-tuning stage hyperparameters
DEFAULT_NUM_MODELS = 10
DEFAULT_EPOCHS = 300
DEFAULT_LR = 4e-4          # [Extension] reused from base stage; paper doesn't restate for fine-tuning
DEFAULT_WEIGHT_DECAY = 5e-5  # [Extension] reused from base stage; paper doesn't restate for fine-tuning
DEFAULT_BATCH_SIZE = 32
BASE_SPLIT_SEED = 42  # must match train_base.py, so each model reuses its original split


def train_one_cae_model(
    model_index: int,
    num_reagents: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    use_wandb: bool,
) -> dict:
    """Fine-tunes a single ensemble member on soft labels and saves its CAE checkpoint."""

    # Reproduce this model's exact original train/val split from train_base.py.
    train_loader, val_loader, _, reagent_vocab = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        split_seed=BASE_SPLIT_SEED + model_index,
    )
    assert len(reagent_vocab) == num_reagents, (
        f"Reagent vocabulary size mismatch: expected {num_reagents}, got {len(reagent_vocab)}"
    )

    # [Base Paper] Load the corresponding trained base-ensemble model as the starting point.
    base_checkpoint_path = os.path.join(BASE_CHECKPOINT_DIR, f"model_{model_index}.pt")
    if not os.path.exists(base_checkpoint_path):
        raise FileNotFoundError(
            f"Base ensemble checkpoint not found: {base_checkpoint_path}. "
            f"Run train_base.py first to produce this checkpoint."
        )
    base_checkpoint = torch.load(base_checkpoint_path, map_location=device)

    model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=False, dropout=0.2).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()  # supports soft float targets in [0, 1] directly

    if use_wandb:
        import wandb
        wandb.init(
            project="medical-ai-cae-ensemble",
            name=f"cae_model_{model_index}",
            config={
                "model_index": model_index,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "split_seed": BASE_SPLIT_SEED + model_index,
                "fine_tuned_from": base_checkpoint_path,
            },
            reinit=True,
        )

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(CAE_CHECKPOINT_DIR, f"model_{model_index}.pt")

    for epoch in range(epochs):
        # --- Training on soft labels ---
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        for images, reagents, _hard_labels, soft_labels in train_loader:
            images, reagents, soft_labels = images.to(device), reagents.to(device), soft_labels.to(device)

            optimizer.zero_grad()
            logits = model(images, reagents)
            loss = criterion(logits, soft_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_total += images.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_total

        # --- Validation: soft-label loss (primary) + hard-label accuracy (monitoring) ---
        model.eval()
        val_loss_sum = 0.0
        val_correct_hard = 0
        val_total = 0

        with torch.no_grad():
            for images, reagents, hard_labels, soft_labels in val_loader:
                images = images.to(device)
                reagents = reagents.to(device)
                hard_labels = hard_labels.to(device)
                soft_labels = soft_labels.to(device)

                logits = model(images, reagents)
                loss = criterion(logits, soft_labels)

                val_loss_sum += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct_hard += (preds == hard_labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_hard_acc = val_correct_hard / val_total

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_soft_loss": train_loss,
                "val_soft_loss": val_loss,
                "val_hard_acc": val_hard_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(
                f"  [CAE Model {model_index}] Epoch {epoch + 1}/{epochs} | "
                f"train_soft_loss={train_loss:.4f} | "
                f"val_soft_loss={val_loss:.4f} val_hard_acc={val_hard_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # [Base Paper] Save final-epoch weights, consistent with the fixed-epoch procedure
    # used for the base ensemble stage (no stated early stopping in the paper).
    os.makedirs(CAE_CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "reagent_vocab": reagent_vocab,
        "model_index": model_index,
        "final_val_soft_loss": val_loss,
        "final_val_hard_acc": val_hard_acc,
        "best_val_soft_loss": best_val_loss,
        "fine_tuned_from": base_checkpoint_path,
    }, checkpoint_path)

    if use_wandb:
        import wandb
        wandb.finish()

    return {
        "model_index": model_index,
        "final_val_soft_loss": val_loss,
        "final_val_hard_acc": val_hard_acc,
        "checkpoint_path": checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the CAE (soft-label) ensemble.")
    parser.add_argument("--num_models", type=int, default=DEFAULT_NUM_MODELS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Fine-tuning {args.num_models} models for {args.epochs} epochs each.\n")

    _, _, _, reagent_vocab = get_dataloaders(batch_size=args.batch_size, split_seed=BASE_SPLIT_SEED)
    num_reagents = len(reagent_vocab)
    print(f"Reagent vocabulary size: {num_reagents}\n")

    results = []
    overall_start = time.time()

    for i in range(args.num_models):
        print(f"--- Fine-tuning CAE model {i + 1}/{args.num_models} ---")
        model_start = time.time()

        result = train_one_cae_model(
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
        print(f"  CAE Model {i} done in {elapsed:.1f}s | "
              f"final_val_hard_acc={result['final_val_hard_acc']:.4f}\n")

    total_elapsed = time.time() - overall_start
    print(f"=== CAE ensemble fine-tuning complete in {total_elapsed / 60:.1f} minutes ===")
    for r in results:
        print(f"  Model {r['model_index']}: final_val_soft_loss={r['final_val_soft_loss']:.4f}, "
              f"final_val_hard_acc={r['final_val_hard_acc']:.4f}, saved to {r['checkpoint_path']}")

    avg_val_hard_acc = sum(r["final_val_hard_acc"] for r in results) / len(results)
    print(f"\nAverage final val hard-label accuracy across CAE ensemble: {avg_val_hard_acc:.4f}")


if __name__ == "__main__":
    main()