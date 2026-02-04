"""
End-to-end example of training a simple computer vision classifier on CIFAR-10.

This script:
- downloads CIFAR-10,
- builds dataloaders with standard ImageNet-style preprocessing,
- trains a ResNet-18 from scratch,
- tracks loss/accuracy per epoch,
- and finally computes confusion matrix + precision/recall/F1 on the validation set.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Config:
    """
    Simple configuration container for all hyperparameters and paths.

    You can change values here instead of editing the code everywhere.
    """

    # Base folder where all script artifacts will be stored
    base_dir: str = "./cursor_play_files"
    # Where CIFAR-10 will be downloaded / read from
    data_dir: str = "./cursor_play_files/data"
    # Mini-batch size used for both training and validation
    batch_size: int = 128
    # Number of passes over the full training set
    num_epochs: int = 12
    # Learning rate for the optimizer
    lr: float = 1e-3
    # Number of worker processes for data loading
    num_workers: int = 4
    # CIFAR-10 has 10 classes (airplane, car, bird, ...)
    num_classes: int = 10
    # Folder for model checkpoints
    checkpoints_dir: str = "./cursor_play_files/checkpoints"
    # Folder for metrics (confusion matrix, classification report, etc.)
    metrics_dir: str = "./cursor_play_files/metrics"
    # Print training progress after this many steps
    print_every: int = 100
    # Use GPU if available, otherwise CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ---------------------------
# Dataset & Dataloaders
# ---------------------------
# Training-time augmentation and normalization. We:
# - randomly crop / resize to 224x224,
# - randomly flip horizontally,
# - convert to tensor,
# - normalize with ImageNet stats (works fine for CIFAR-10 too).
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# Validation-time preprocessing (no random augmentation, only resize + center crop).
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# CIFAR-10 training split
train_dataset = datasets.CIFAR10(
    root=cfg.data_dir,
    train=True,
    transform=train_transform,
    download=True,
)

# CIFAR-10 validation (test) split
val_dataset = datasets.CIFAR10(
    root=cfg.data_dir,
    train=False,
    transform=val_transform,
    download=True,
)

# DataLoader shuffles training data every epoch
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

# Validation loader does NOT shuffle; we just iterate once per epoch
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)


# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
def build_model(num_classes: int) -> nn.Module:
    """
    Build a ResNet-18 and replace the final fully-connected layer
    to match the number of target classes.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# Instantiate model, loss function, optimizer and LR scheduler
model = build_model(cfg.num_classes).to(cfg.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.2)


# ---------------------------
# Train / Eval loops
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int):
    """
    Run one full training epoch.

    Args:
        model: the neural network being trained.
        loader: DataLoader for the training set.
        optimizer: optimizer (e.g. Adam).
        criterion: loss function (e.g. cross-entropy).
        device: "cuda" or "cpu".
        epoch: current epoch index (1-based, only for logging).

    Returns:
        (epoch_loss, epoch_acc): average loss and accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

        if (step + 1) % cfg.print_every == 0:
            print(
                f"Epoch [{epoch}] Step [{step+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a given dataset (no gradient computation).

    Args:
        model: trained model.
        loader: DataLoader for validation/test.
        criterion: loss function.
        device: "cuda" or "cpu".

    Returns:
        (epoch_loss, epoch_acc): average loss and accuracy on the dataset.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def compute_metrics_and_confusion_matrix(model, loader, device):
    """
    Compute precision, recall, F1 (macro) and confusion matrix on a dataset.

    This function collects all predictions / labels, then:
    - prints macro precision / recall / F1,
    - prints a per-class classification report,
    - saves a confusion matrix heatmap as PNG.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        _, preds = outputs.max(1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm = confusion_matrix(all_targets, all_preds)
    macro_precision = precision_score(all_targets, all_preds, average="macro")
    macro_recall = recall_score(all_targets, all_preds, average="macro")
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    print("Macro Precision:", macro_precision)
    print("Macro Recall   :", macro_recall)
    print("Macro F1       :", macro_f1)

    # Text classification report for quick inspection in the terminal
    print("\nClassification report (per class):")
    report_text = classification_report(all_targets, all_preds)
    print(report_text)

    # Save classification report as a matrix (CSV) for later analysis
    report_dict = classification_report(all_targets, all_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Ensure output directory exists
    os.makedirs(cfg.metrics_dir, exist_ok=True)

    report_df.to_csv(os.path.join(cfg.metrics_dir, "classification_report_val.csv"), index=True)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Validation")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.metrics_dir, "confusion_matrix_val.png"))
    plt.close()


def main():
    """
    Entry point for training and evaluation.

    Trains the model for cfg.num_epochs, saves the best checkpoint based on
    validation accuracy, then computes detailed metrics and confusion matrix.
    """
    best_val_acc = 0.0
    # Create base folders for data/checkpoints/metrics if they don't exist
    os.makedirs(cfg.base_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)

        print(
            f"Epoch {epoch}/{cfg.num_epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                },
                os.path.join(cfg.checkpoints_dir, "best_resnet18.pt"),
            )
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

        # Step the learning rate scheduler once per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate after epoch {epoch}: {current_lr:.6f}")

    print(f"Training done. Best val_acc={best_val_acc:.4f}")

    # Final metrics and confusion matrix on validation set
    print("\nComputing validation metrics and confusion matrix...")
    compute_metrics_and_confusion_matrix(model, val_loader, cfg.device)


if __name__ == "__main__":
    main()



