"""
utils.py

Shared utilities for the Adaptive Reliability-Aware Confidence Learning project.
Currently contains the model architecture shared by train_base.py (hard-label
ensemble, epistemic uncertainty) and train_cae.py (soft-label fine-tuned ensemble,
aleatoric uncertainty) — per the base paper, both ensembles use the same K models,
just fine-tuned on a different label type in the second stage.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Reagent(nn.Module):
    """
    [Base Paper, Section 5.2] MobileNet-V3-small backbone, ImageNet-pretrained, with a
    one-hot reagent encoding injected at the dense classifier layers, and a dropout
    layer included as one of the paper's stated sources of inter-model ensemble
    diversity (alongside different data augmentations, initializations, and random
    train/val splits, which are handled elsewhere in the training pipeline).

    Outputs a single logit for binary classification (positive/negative agglutination
    reaction). Apply torch.sigmoid() externally to get a probability in [0, 1].
    """

    def __init__(self, num_reagents: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Keep the convolutional feature extractor and pooling layer from the backbone;
        # discard its original 1000-class ImageNet classifier head, which we replace below.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone_out_dim = backbone.classifier[0].in_features  # 576 for MobileNet-V3-small

        # [Base Paper] "a dropout layer" — explicitly listed as a diversity source across
        # the K ensemble members, since each model gets independently-sampled dropout.
        self.dropout_layer = nn.Dropout(p=dropout)

        # [Base Paper] one-hot reagent encoding "through one-hot encoding at one of the
        # neural network's dense layers" — concatenated here, at the first dense layer
        # after the pooled visual features.
        self.fc1 = nn.Linear(backbone_out_dim + num_reagents, 1024)
        self.act1 = nn.Hardswish()
        self.fc2 = nn.Linear(1024, 1)

    def get_embedding(self, image: torch.Tensor, reagent_onehot: torch.Tensor) -> torch.Tensor:
        """
        [Extension, Phase 2] Returns the pre-classifier feature vector (pooled visual
        features concatenated with the reagent one-hot encoding), used as the embedding
        space for the reliability estimator's nearest-neighbor lookups. Shape:
        (batch, backbone_out_dim + num_reagents) -- 587-dim for this architecture.
        Deliberately taken *before* the dropout layer, so embeddings are deterministic
        (not affected by dropout's random masking) regardless of train/eval mode.
        """
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, reagent_onehot], dim=1)
        return x

    def forward(self, image: torch.Tensor, reagent_onehot: torch.Tensor) -> torch.Tensor:
        """
        image: (batch, 3, 224, 224)
        reagent_onehot: (batch, num_reagents)
        returns: (batch,) raw logits (apply sigmoid externally for probabilities)
        """
        x = self.get_embedding(image, reagent_onehot)
        x = self.dropout_layer(x)
        x = self.fc1(x)
        x = self.act1(x)
        logit = self.fc2(x)
        return logit.squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_reagents = 11  # matches the vocabulary size confirmed from data_loader.py
    model = MobileNetV3Reagent(num_reagents=num_reagents, pretrained=True).to(device)

    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Dummy forward pass to confirm shapes and measure VRAM usage.
    batch_size = 32
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_reagents = torch.zeros(batch_size, num_reagents).to(device)
    dummy_reagents[:, 0] = 1.0  # one-hot: all samples assigned to reagent index 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    output = model(dummy_images, dummy_reagents)
    print(f"Output shape: {output.shape} (expected: torch.Size([{batch_size}]))")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory used (forward pass only): {peak_mb:.1f} MB")

    probs = torch.sigmoid(output)
    print(f"Sample output probabilities: {probs[:5].detach().cpu().tolist()}")