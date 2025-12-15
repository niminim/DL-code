import os
import time
import copy
import random
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

import matplotlib
matplotlib.use("Qt5Agg")  # Use Qt5Agg backend (works on your system)
import matplotlib.pyplot as plt


# -----------------------------
# Config & Paths
# -----------------------------
DATA_ROOT = "/home/nim/Downloads/Data/OCT2017"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

RESULTS_DIR = "/home/nim/venv/DL-code/Classification/OCT_Cls_v2/training_processs/results"
GRAPHS_DIR = "/home/nim/venv/DL-code/Classification/OCT_Cls_v2/training_processs/graphs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

NUM_TRAIN_SAMPLES = 200
NUM_VAL_SAMPLES = 500
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # early stopping patience (epochs without val acc improvement)
SEED = 42
NUM_CLASSES = 4
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Matplotlib global style (readable plots)
# -----------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# -----------------------------
# Data transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -----------------------------
# Datasets & Dataloaders
# -----------------------------
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
print(f"Total images in original train folder: {len(full_train_dataset)}")
print("Classes in dataset:", full_train_dataset.classes)

if len(full_train_dataset) < NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES:
    raise ValueError(
        f"Requested {NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES} images (train+val) "
        f"but dataset has only {len(full_train_dataset)}"
    )

indices = list(range(len(full_train_dataset)))
np.random.shuffle(indices)
selected_indices = indices[:NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES]
train_indices = selected_indices[:NUM_TRAIN_SAMPLES]
val_indices = selected_indices[NUM_TRAIN_SAMPLES:]

train_dataset = Subset(full_train_dataset, train_indices)

# Validation uses eval transforms (no augmentation)
full_train_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
val_dataset = Subset(full_train_eval, val_indices)

test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

dataloaders = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader
}


# -----------------------------
# Model: MobileNetV3
# -----------------------------
def create_mobilenetv3(num_classes: int):
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_large(pretrained=True)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


model = create_mobilenetv3(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# -----------------------------
# Metrics helper
# -----------------------------
def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    return acc, precision, recall, f1


def fmt(x):
    """Format float to 4 decimals as string."""
    return f"{x:.4f}"


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, precision, recall, f1 = compute_metrics(all_labels, all_preds)

    return epoch_loss, acc, precision, recall, f1, np.array(all_labels), np.array(all_preds)


# -----------------------------
# Training loop with early stopping
# -----------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, patience, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    epochs_since_improvement = 0

    history = {
        "train": {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []},
        "val": {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []},
        "test": {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []},
    }

    csv_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "split", "loss", "acc", "precision", "recall", "f1"])

    for epoch in range(num_epochs):
        print("-" * 60)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # -----------------
        # Train phase
        # -----------------
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(dataloaders["train"].dataset)
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(all_labels, all_preds)

        history["train"]["loss"].append(train_loss)
        history["train"]["acc"].append(train_acc)
        history["train"]["precision"].append(train_prec)
        history["train"]["recall"].append(train_rec)
        history["train"]["f1"].append(train_f1)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1, "train",
                    fmt(train_loss), fmt(train_acc),
                    fmt(train_prec), fmt(train_rec), fmt(train_f1)
                ]
            )

        print(
            f"  Train   - Loss: {fmt(train_loss)} | "
            f"Acc: {fmt(train_acc)} | Prec: {fmt(train_prec)} | "
            f"Rec: {fmt(train_rec)} | F1: {fmt(train_f1)}"
        )

        # -----------------
        # Validation + Test
        # -----------------
        for split in ["val", "test"]:
            loss, acc, prec, rec, f1, _, _ = evaluate(
                model, dataloaders[split], criterion, device
            )

            history[split]["loss"].append(loss)
            history[split]["acc"].append(acc)
            history[split]["precision"].append(prec)
            history[split]["recall"].append(rec)
            history[split]["f1"].append(f1)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1, split,
                        fmt(loss), fmt(acc),
                        fmt(prec), fmt(rec), fmt(f1)
                    ]
                )

            print(
                f"  {split.capitalize():<7}- Loss: {fmt(loss)} | "
                f"Acc: {fmt(acc)} | Prec: {fmt(prec)} | "
                f"Rec: {fmt(rec)} | F1: {fmt(f1)}"
            )

        # LR scheduler step
        scheduler.step()

        # Early stopping on validation accuracy
        val_acc = history["val"]["acc"][-1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_since_improvement = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            # Rolling "best so far" file
            torch.save(
                best_model_wts,
                os.path.join(RESULTS_DIR, "best_model_state_dict.pth")
            )
        else:
            epochs_since_improvement += 1

        print(f"  Current best val acc: {fmt(best_val_acc)} (epoch {best_epoch})")
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without "
                  "val acc improvement.")
            break

    time_elapsed = time.time() - since
    print("-" * 60)
    print(f"Training complete in {time_elapsed / 60:.1f} minutes")
    print(f"Best val Acc: {fmt(best_val_acc)} (epoch {best_epoch})")

    model.load_state_dict(best_model_wts)
    return model, history, best_epoch, best_val_acc


# -----------------------------
# Plot training curves
# -----------------------------
def plot_training_curves(history, output_path, show=False):
    epochs = range(1, len(history["train"]["loss"]) + 1)

    plt.style.use("default")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # Loss (left axis)
    ax1.plot(epochs, history["train"]["loss"], label="Train Loss",
             linestyle="-", marker="o", linewidth=2, color="blue")
    ax1.plot(epochs, history["val"]["loss"], label="Val Loss",
             linestyle="--", marker="o", linewidth=2, color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train"]["acc"], label="Train Acc",
             linestyle="-", marker="s", linewidth=2, color="green")
    ax2.plot(epochs, history["val"]["acc"], label="Val Acc",
             linestyle="--", marker="s", linewidth=2, color="red")
    ax2.set_ylabel("Accuracy", color="black")
    ax2.tick_params(axis='y', labelcolor='black')

    # Combined legend
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, lbl = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(lbl)
    ax1.legend(lines, labels, loc="lower right")

    plt.title("Training & Validation Loss / Accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Confusion matrix & report images
# -----------------------------
def save_confusion_matrix(y_true, y_pred, class_names, output_path, show=False):
    cm = confusion_matrix(y_true, y_pred)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Blue → White colormap ---
    cmap = plt.cm.Blues

    # --- Show matrix ---
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # --- Colorbar ---
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    # --- Axis labels & ticks ---
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # --- Annotate cells with readable text ---
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j, i, f"{value}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if value > thresh else "black"
            )

    # --- Grid lines for clarity ---
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- SAVE FIGURE ---
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    # --- SHOW FIGURE IF ASKED ---
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_classification_report_image(y_true, y_pred, class_names, output_path, show=False):
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "Classification Report (Best Model on Test Set)\n\n" + report_str,
        fontsize=10,
        va="top",
        family="monospace"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    best_model, history, best_epoch, best_val_acc = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device
    )

    # 1) Training curves: save + show
    curves_path = os.path.join(GRAPHS_DIR, "training_loss_accuracy.png")
    plot_training_curves(history, curves_path, show=True)
    print(f"Saved training curves to: {curves_path}")

    # 2) Best model on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(
        best_model, dataloaders["test"], criterion, device
    )
    print("\n=== Best Model on Full Test Set ===")
    print(
        f"Loss: {fmt(test_loss)} | Acc: {fmt(test_acc)} | "
        f"Prec: {fmt(test_prec)} | Rec: {fmt(test_rec)} | F1: {fmt(test_f1)}"
    )
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=test_dataset.classes, digits=4
    ))

    # 3) Confusion matrix: save + show
    cm_path = os.path.join(GRAPHS_DIR, "confusion_matrix_best_model.png")
    save_confusion_matrix(y_true, y_pred, test_dataset.classes, cm_path, show=True)
    print(f"\nSaved confusion matrix image to: {cm_path}")

    # 4) Classification report image: save + show
    cr_path = os.path.join(GRAPHS_DIR, "classification_report_best_model.png")
    save_classification_report_image(y_true, y_pred, test_dataset.classes, cr_path, show=True)
    print(f"Saved classification report image to: {cr_path}")

    # Save best model with epoch + best val in filename
    best_val_str = fmt(best_val_acc)
    final_model_filename = f"epoch_{best_epoch}_best_val_{best_val_str}.pth"
    final_model_path = os.path.join(RESULTS_DIR, final_model_filename)

    torch.save(best_model.state_dict(), final_model_path)
    print(f"Saved best model weights to: {final_model_path}")


if __name__ == "__main__":
    main()
