import os
import torch
import torchvision.transforms.functional as F
from PIL import Image


def letterbox_image(img, final_size=(224, 224)):
    """
    Resizes 'img' so that the longer side matches 'final_size',
    then pads the other dimension to exactly match 'final_size'.

    Preserves the original aspect ratio (no distortion).
    Returns a PIL Image.
    """
    w, h = img.size  # Original width and height
    target_w, target_h = final_size

    # Figure out the scale so the image's longest side matches final_size's dimension
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # 1) Proportionally resize
    img_resized = F.resize(img, (new_h, new_w))

    # 2) Pad to final dimensions
    # left, top, right, bottom
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    img_padded = F.pad(
        img_resized,
        padding=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0,  # black padding
        padding_mode='constant'
    )

    return img_padded


class LetterboxTransform:
    """
    A custom transform class that letterboxes each image
    to the specified size, preserving aspect ratio.
    """

    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, img):
        # img is a PIL Image
        return letterbox_image(img, self.size)


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system

# Suppose you have an image with unusual aspect ratio
img_path = "/home/nim/Downloads/Data/OCT2017/train/DRUSEN/DRUSEN-66861-1.jpeg"
img_path = "/home/nim/Downloads/Data/OCT2017/train/DRUSEN/DRUSEN-224974-10.jpeg"

original_img = Image.open(img_path)

# Apply letterbox
letterbox = LetterboxTransform(size=(224, 224))
letterboxed_img = letterbox(original_img)

# Show side by side
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original_img)

plt.subplot(1, 2, 2)
plt.title("Letterboxed to 224x224")
plt.imshow(letterboxed_img)

plt.show()