"""
probe_or_finetune.py

IDE-friendly (no argparse): Linear probe / Fine-tune for a MoCo v2 checkpoint.

Folder layout (under BASE_OUT_DIR):

pretraining/
  <PRETRAIN_ID>/
    linear_eval/
      <RUN_NAME>/
        results/
        graphs/
        models/
    finetuning/
      <RUN_NAME>/
        results/
        graphs/
        models/

Requested:
- Add option to choose number of labeled training images (NUM_TRAIN_LABELED)
- Save best model in models/
- Final model filename like:
    mobilenetv3_epoch_9_val_acc_0.9420.pth
  (i.e., <backbone>_epoch_<best_epoch>_val_acc_<best_val_acc>.pth)
"""

import time
import copy
import random
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from torch import amp


# ============================================================
# IDE RUN CONFIG (EDIT THESE)
# ============================================================
DATA_ROOT = Path("/home/nim/Downloads/Data/OCT2017")

CKPT_PATH = Path("/home/nim/venv/DL-code/Classification/OCT_Cls_v2/mocov2_checkpoints/moco_v2_epoch_140_loss_5.9698.pt")

MODE = "finetune"  # "linear" or "finetune"

NUM_EPOCHS = 20
PATIENCE = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
SEED = 42

# Limit how many labeled images are used for training (None = use all)
NUM_TRAIN_LABELED = 200  # e.g. 200, 1000, 5000

SHOW_PLOTS = True

BASE_OUT_DIR = Path("/home/nim/venv/DL-code/Classification/OCT_Cls_v2")
# ============================================================


UNSUP_DIR = DATA_ROOT / "data_for_unsupervised"
TRAIN_LABELED = UNSUP_DIR / "train_labeled"
TEST_OUT = UNSUP_DIR / "test"


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


# -----------------------------
# Transforms
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
# Run dir helper (no overwrite)
# -----------------------------
def make_run_dir(base_dir: Path, ckpt_path: Path, stage: str, mode: str, bs: int, lr: float, seed: int) -> Path:
    pretrain_id = ckpt_path.stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{mode}_bs{bs}_lr{lr}_seed{seed}_{stamp}"
    return base_dir / "pretraining" / pretrain_id / stage / run_name


# -----------------------------
# Metrics helpers
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, precision, recall, f1


def fmt(x):
    return f"{x:.4f}"


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_sum = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    loss = loss_sum / len(dataloader.dataset)
    acc, p, r, f1 = compute_metrics(all_labels, all_preds)
    return loss, acc, p, r, f1, np.array(all_labels), np.array(all_preds)


# -----------------------------
# Backbone builders
# -----------------------------
def build_resnet18_backbone():
    net = models.resnet18(weights=None)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim, "resnet18"


def build_mobilenetv3_backbone():
    net = models.mobilenet_v3_large(weights=None)
    feat_dim = net.classifier[-1].in_features
    net.classifier = nn.Identity()
    return net, feat_dim, "mobilenetv3"


def build_backbone(name: str):
    if name == "resnet18":
        return build_resnet18_backbone()
    if name in ("mobilenetv3", "mobilenet_v3_large"):
        return build_mobilenetv3_backbone()
    raise ValueError(f"Unknown backbone: {name}")


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# -----------------------------
# MoCo checkpoint loader
# -----------------------------
def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_encoder_q_into_backbone(ckpt: dict, backbone: nn.Module):
    # Case 1: ckpt["encoder_q"] is directly the backbone state_dict
    if isinstance(ckpt, dict) and "encoder_q" in ckpt and isinstance(ckpt["encoder_q"], dict):
        backbone.load_state_dict(ckpt["encoder_q"], strict=True)
        return

    # Case 2: ckpt["full_model"] contains keys like "encoder_q.layer1.0..."
    if isinstance(ckpt, dict) and "full_model" in ckpt and isinstance(ckpt["full_model"], dict):
        enc_sd = _strip_prefix(ckpt["full_model"], "encoder_q.")
        if not enc_sd:
            raise KeyError("ckpt['full_model'] exists but contains no 'encoder_q.' keys")
        backbone.load_state_dict(enc_sd, strict=True)
        return

    # Case 3: checkpoint itself is a full model state dict with "encoder_q."
    if isinstance(ckpt, dict) and any(k.startswith("encoder_q.") for k in ckpt.keys()):
        enc_sd = _strip_prefix(ckpt, "encoder_q.")
        backbone.load_state_dict(enc_sd, strict=True)
        return

    raise ValueError("Cannot find encoder_q weights in checkpoint")


# -----------------------------
# Plot + reports (optional, minimal)
# -----------------------------
def save_confusion_matrix_image(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Training loop (keeps best checkpoint in models/)
# -----------------------------
def train_model(model, loaders, criterion, optimizer, scheduler,
                epochs, patience, device, results_dir: Path, models_dir: Path, backbone_tag: str):
    best_val_acc = 0.0
    best_epoch = 0
    best_wts = copy.deepcopy(model.state_dict())
    since_best = 0

    # CSV
    csv_path = results_dir / "training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "loss", "acc", "precision", "recall", "f1"])

    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- Train
        model.train()
        loss_sum = 0.0
        all_preds, all_labels = [], []

        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            all_preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

        train_loss = loss_sum / len(loaders["train"].dataset)
        train_acc, train_p, train_r, train_f1 = compute_metrics(all_labels, all_preds)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", fmt(train_loss), fmt(train_acc), fmt(train_p), fmt(train_r), fmt(train_f1)])

        # ---- Val
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(model, loaders["val"], criterion, device)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "val", fmt(val_loss), fmt(val_acc), fmt(val_p), fmt(val_r), fmt(val_f1)])

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        # ---- Track best + save rolling best in models/
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())
            since_best = 0

            torch.save(best_wts, models_dir / f"{backbone_tag}_best_running.pth")
        else:
            since_best += 1

        if since_best >= patience:
            print(f"Early stopping after {patience} epochs without val acc improvement.")
            break

    model.load_state_dict(best_wts)
    return model, best_epoch, best_val_acc


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not TRAIN_LABELED.exists():
        raise FileNotFoundError(f"TRAIN_LABELED not found: {TRAIN_LABELED}")
    if not TEST_OUT.exists():
        raise FileNotFoundError(f"TEST_OUT not found: {TEST_OUT}")

    stage = "linear_eval" if MODE == "linear" else "finetuning"
    run_dir = make_run_dir(BASE_OUT_DIR, CKPT_PATH, stage, MODE, BATCH_SIZE, LEARNING_RATE, SEED)

    results_dir = run_dir / "results"
    graphs_dir = run_dir / "graphs"
    models_dir = run_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OUT] Writing to: {run_dir}")

    # -----------------------------
    # Datasets
    # -----------------------------
    labeled_train_full = datasets.ImageFolder(TRAIN_LABELED, transform=train_transform)
    class_names = labeled_train_full.classes
    num_classes = len(class_names)

    indices = np.random.permutation(len(labeled_train_full))
    n_val = max(1, int(len(indices) * VAL_RATIO))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if NUM_TRAIN_LABELED is not None:
        if NUM_TRAIN_LABELED <= 0:
            raise ValueError("NUM_TRAIN_LABELED must be positive or None.")
        if NUM_TRAIN_LABELED > len(train_idx):
            raise ValueError(
                f"NUM_TRAIN_LABELED={NUM_TRAIN_LABELED} > available train samples: {len(train_idx)}"
            )
        train_idx = train_idx[:NUM_TRAIN_LABELED]

    print(f"Train labeled samples: {len(train_idx)} | Val samples: {len(val_idx)}")

    train_ds = Subset(labeled_train_full, train_idx)

    labeled_eval_full = datasets.ImageFolder(TRAIN_LABELED, transform=eval_transform)
    val_ds = Subset(labeled_eval_full, val_idx)

    test_ds = datasets.ImageFolder(TEST_OUT, transform=eval_transform)

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
    }

    # -----------------------------
    # Load checkpoint + build backbone
    # -----------------------------
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    backbone_name = "resnet18"
    if isinstance(ckpt, dict) and "backbone" in ckpt:
        backbone_name = ckpt["backbone"]

    backbone, feat_dim, backbone_tag = build_backbone(backbone_name)
    load_encoder_q_into_backbone(ckpt, backbone)

    model = BackboneWithHead(backbone, feat_dim, num_classes).to(device)

    if MODE == "linear":
        set_requires_grad(model.backbone, False)
        set_requires_grad(model.head, True)
        print("Mode: linear probe (backbone frozen)")
    else:
        set_requires_grad(model.backbone, True)
        set_requires_grad(model.head, True)
        print("Mode: finetune (backbone + head trainable)")

    # -----------------------------
    # Train
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_model, best_epoch, best_val_acc = train_model(
        model, loaders, criterion, optimizer, scheduler,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
        results_dir=results_dir,
        models_dir=models_dir,
        backbone_tag=backbone_tag,
    )

    # -----------------------------
    # Evaluate best on test + save artifacts
    # -----------------------------
    test_loss, test_acc, test_p, test_r, test_f1, y_true, y_pred = evaluate(
        best_model, loaders["test"], criterion, device
    )

    print("\n=== Best Model on Full Test Set ===")
    print(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Prec: {test_p:.4f} | Rec: {test_r:.4f} | F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4))

    # Confusion matrix image
    cm_path = graphs_dir / "confusion_matrix_best_model.png"
    save_confusion_matrix_image(y_true, y_pred, test_ds.classes, cm_path)
    print(f"Saved confusion matrix image to: {cm_path}")

    # -----------------------------
    # Save final best model weights in required filename format
    # -----------------------------
    final_name = f"{backbone_tag}_epoch_{best_epoch}_val_acc_{best_val_acc:.4f}.pth"
    final_path = models_dir / final_name
    torch.save(best_model.state_dict(), final_path)
    print(f"Saved best model weights to: {final_path}")


if __name__ == "__main__":
    main()
