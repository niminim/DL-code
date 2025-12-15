import os
import time
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report
from torch import amp


# -----------------------------
# Paths
# -----------------------------
DATA_ROOT = Path("/home/nim/Downloads/Data/OCT2017")
UNSUP_DIR = DATA_ROOT / "data_for_unsupervised"
TRAIN_LABELED = UNSUP_DIR / "train_labeled"
TEST_OUT = UNSUP_DIR / "test"

OUT_DIR = Path("/home/nim/venv/DL-code/SSL_OCT_MoCoV2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
# Backbone builder (must match SSL file)
# -----------------------------
def build_backbone(name: str):
    if name == "resnet18":
        net = models.resnet18(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=None)
        feat_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    raise ValueError(f"Unknown backbone: {name}")


class LinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    return acc, all_labels, all_preds


def train(model, train_loader, test_loader, device, epochs, lr, weight_decay):
    criterion = nn.CrossEntropyLoss()

    # Only parameters with requires_grad=True will be optimized (works for freeze/finetune)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        avg_loss = running / len(train_loader)
        acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} | test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MoCo checkpoint (.pt)")
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "finetune"],
                        help="linear=freeze backbone; finetune=train backbone+head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)            # good for Adam probe
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Data
    train_ds = datasets.ImageFolder(TRAIN_LABELED, transform=train_transform)
    test_ds = datasets.ImageFolder(TEST_OUT, transform=eval_transform)
    print("Train labeled:", len(train_ds), "classes:", train_ds.classes)
    print("Test:", len(test_ds), "classes:", test_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone_name = ckpt.get("backbone", "resnet18")
    print("Checkpoint backbone:", backbone_name)

    backbone, feat_dim = build_backbone(backbone_name)
    backbone.load_state_dict(ckpt["encoder_q"], strict=True)

    model = LinearProbe(backbone, feat_dim, args.num_classes).to(device)

    # Freeze or finetune
    if args.mode == "linear":
        set_requires_grad(model.backbone, False)   # freeze
        set_requires_grad(model.head, True)        # train head
        print("Mode: linear probe (backbone frozen)")
    else:
        set_requires_grad(model.backbone, True)
        set_requires_grad(model.head, True)
        print("Mode: finetune (backbone + head trainable)")

    # Train
    model, best_acc = train(model, train_loader, test_loader, device, args.epochs, args.lr, args.wd)

    # Final eval + report
    acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("\nFinal Test Acc:", f"{acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=test_ds.classes, digits=4))

    # Save
    tag = f"{args.mode}_epochs{args.epochs}_acc{best_acc:.4f}"
    out_path = OUT_DIR / f"probe_{tag}.pth"
    torch.save(model.state_dict(), out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
