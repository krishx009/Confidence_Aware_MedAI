"""
evaluate_base_ensemble_testacc.py

Quick verification script: loads all K=10 base ensemble checkpoints and evaluates each
one's hard-label accuracy on the held-out test set (2,516 images). This is the number
directly comparable to the base paper's reported result (Section 5.4): "Ten neural
networks of a classification ensemble achieved an accuracy of 0.9671 ± 0.0058 on the
test subset."
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataloaders
from utils import MobileNetV3Reagent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "base_ensemble")
NUM_MODELS = 10


def evaluate_model_on_test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, reagents, hard_labels, _soft_labels in test_loader:
            images, reagents, hard_labels = images.to(device), reagents.to(device), hard_labels.to(device)
            logits = model(images, reagents)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == hard_labels).sum().item()
            total += images.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Test set is identical regardless of split_seed (it's not affected by the
    # train/val stratified split), so any split_seed value here is fine.
    _, _, test_loader, reagent_vocab = get_dataloaders(batch_size=32, split_seed=42)
    num_reagents = len(reagent_vocab)

    accuracies = []
    for i in range(NUM_MODELS):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{i}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_acc = evaluate_model_on_test(model, test_loader, device)
        accuracies.append(test_acc)
        print(f"Model {i}: test accuracy = {test_acc:.4f}")

    accuracies_t = torch.tensor(accuracies)
    print(f"\nMean test accuracy: {accuracies_t.mean().item():.4f}")
    print(f"Std dev:             {accuracies_t.std().item():.4f}")
    print(f"\nBase paper reports:  0.9671 ± 0.0058 (Section 5.4)")


if __name__ == "__main__":
    main()