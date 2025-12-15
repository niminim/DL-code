import os
import time
import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import amp


# -----------------------------
# Paths (your structure)
# -----------------------------
DATA_ROOT = Path("/home/nim/Downloads/Data/OCT2017")

UNSUP_DIR = DATA_ROOT / "data_for_unsupervised"
TRAIN_UNLABELED = UNSUP_DIR / "train_unlabeled"

OUT_DIR = Path("/home/nim/venv/DL-code/SSL_OCT_MoCoV2")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config
# -----------------------------
SEED = 42
IMG_SIZE = 224

BATCH_SIZE = 512
NUM_WORKERS = 16
PREFETCH_FACTOR = 1
EPOCHS = 100

LR = 0.03
WEIGHT_DECAY = 1e-4
SGD_MOMENTUM = 0.9

# MoCo v2
DIM = 128
MLP_HIDDEN = 512
QUEUE_K = 32768      # must be divisible by BATCH_SIZE in this minimal single-GPU impl
MOMENTUM_M = 0.99
TEMPERATURE = 0.2

BACKBONE = "resnet18"  # "mobilenet_v3_large" also supported


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# Two-view augmentation
# -----------------------------
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base = base_transform

    def __call__(self, x):
        return self.base(x), self.base(x)


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))],
        p=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -----------------------------
# Backbone + projection head
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


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def momentum_update(model_q: nn.Module, model_k: nn.Module, m: float):
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)


class MoCoV2(nn.Module):
    def __init__(self, backbone_name: str, dim: int, mlp_hidden: int, K: int, m: float, T: float):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q, feat_dim = build_backbone(backbone_name)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        self.proj_q = MLP(feat_dim, mlp_hidden, dim)
        self.proj_k = copy.deepcopy(self.proj_q)

        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.proj_k.parameters():
            p.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # keys: [N, dim]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if self.K % batch_size != 0:
            raise ValueError(
                f"QUEUE_K ({self.K}) must be divisible by batch size ({batch_size}) in this simple implementation."
            )

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        # Query
        q = self.encoder_q(im_q)
        q = self.proj_q(q)
        q = F.normalize(q, dim=1)

        # Key (momentum)
        with torch.no_grad():
            momentum_update(self.encoder_q, self.encoder_k, self.m)
            momentum_update(self.proj_q, self.proj_k, self.m)

            k = self.encoder_k(im_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=1)

        # logits: [N, 1+K]
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)
        return logits, labels


def cosine_lr(base_lr: float, epoch: int, max_epoch: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch))


def save_ckpt(path: Path, epoch: int, model: MoCoV2, optimizer: torch.optim.Optimizer):
    ckpt = {
        "epoch": epoch,
        "backbone": BACKBONE,
        "encoder_q": model.encoder_q.state_dict(),
        "proj_q": model.proj_q.state_dict(),
        "full_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, path)


def main():
    dataset = datasets.ImageFolder(TRAIN_UNLABELED, transform=TwoCropsTransform(moco_transform))
    print(f"Unlabeled images: {len(dataset)} classes: {dataset.classes}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    model = MoCoV2(
        backbone_name=BACKBONE,
        dim=DIM,
        mlp_hidden=MLP_HIDDEN,
        K=QUEUE_K,
        m=MOMENTUM_M,
        T=TEMPERATURE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        list(model.encoder_q.parameters()) + list(model.proj_q.parameters()),
        lr=LR,
        momentum=SGD_MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()

        lr = cosine_lr(LR, epoch - 1, EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        running = 0.0
        for (x_q, x_k), _ in loader:
            x_q = x_q.to(device, non_blocking=True)
            x_k = x_k.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, labels = model(x_q, x_k)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        avg_loss = running / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{EPOCHS} | lr={lr:.5f} | loss={avg_loss:.4f} | {dt:.1f}s")

        if epoch % 10 == 0 or epoch == EPOCHS:
            ckpt_path = CKPT_DIR / f"moco_v2_{BACKBONE}_epoch_{epoch:03d}.pt"
            save_ckpt(ckpt_path, epoch, model, optimizer)
            print("Saved:", ckpt_path)

    print("Done.")


if __name__ == "__main__":
    main()


# To clear old swap residue before a long run:
# sudo swapoff -a
# sudo swapon -a