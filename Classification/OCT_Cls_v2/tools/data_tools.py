from pathlib import Path
import shutil
import random

# -------------------------------------------------------------------------
# Base directories
# -------------------------------------------------------------------------
DATA_ROOT = Path("/home/nim/Downloads/Data/OCT2017")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

# New dataset structure
UNSUP_DIR = DATA_ROOT / "data_for_unsupervised"
TRAIN_LABELED = UNSUP_DIR / "train_labeled"
TRAIN_UNLABELED = UNSUP_DIR / "train_unlabeled"
TEST_OUT = UNSUP_DIR / "test"


# -------------------------------------------------------------------------
# Helper: copy a directory cleanly (delete if exists)
# -------------------------------------------------------------------------
def copy_clean(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)   # remove old folder
    shutil.copytree(src, dst)


# -------------------------------------------------------------------------
# 1. Ensure directories exist
# -------------------------------------------------------------------------
TRAIN_LABELED.mkdir(parents=True, exist_ok=True)
TRAIN_UNLABELED.mkdir(parents=True, exist_ok=True)
UNSUP_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# 2. Copy test dataset as-is
# -------------------------------------------------------------------------
copy_clean(TEST_DIR, TEST_OUT)
print(f"[OK] Copied test set → {TEST_OUT}")


# -------------------------------------------------------------------------
# 3. Split the training dataset into 5% labeled / 95% unlabeled
# -------------------------------------------------------------------------
# Each class folder (CNV / DME / DRUSEN / NORMAL)
classes = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]

for cls_dir in classes:
    cls_name = cls_dir.name

    # Collect all image paths under this class
    images = [p for p in cls_dir.rglob("*") if p.is_file()]
    random.shuffle(images)

    # Split sizes
    n_total = len(images)
    n_labeled = int(n_total * 0.05)

    labeled_imgs = images[:n_labeled]
    unlabeled_imgs = images[n_labeled:]

    # Output class folders
    labeled_cls_out = TRAIN_LABELED / cls_name
    unlabeled_cls_out = TRAIN_UNLABELED / cls_name
    labeled_cls_out.mkdir(parents=True, exist_ok=True)
    unlabeled_cls_out.mkdir(parents=True, exist_ok=True)

    # Copy labeled images
    for img in labeled_imgs:
        shutil.copy2(img, labeled_cls_out / img.name)

    # Copy unlabeled images
    for img in unlabeled_imgs:
        shutil.copy2(img, unlabeled_cls_out / img.name)

    print(f"[{cls_name}] total={n_total}, labeled={len(labeled_imgs)}, unlabeled={len(unlabeled_imgs)}")


# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
print("\n[ALL DONE] Unsupervised dataset created successfully!")
print(f"Train labeled directory   → {TRAIN_LABELED}")
print(f"Train unlabeled directory → {TRAIN_UNLABELED}")
print(f"Test directory            → {TEST_OUT}")
