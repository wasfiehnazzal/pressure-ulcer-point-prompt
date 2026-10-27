"""
Point-prompted DeepLabV3+ demo for pressure ulcer segmentation.

Runs the 4-channel model (RGB + Gaussian point prompt) on the bundled example
image and compares the versions trained without and with data augmentation.

    python demo.py
"""

import os
import hashlib

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

_HERE = os.path.dirname(os.path.abspath(__file__))

IMAGE = os.path.join(_HERE, "example", "example.jpg")
MASK = os.path.join(_HERE, "example", "example.npy")
WEIGHTS_NOAUG = os.path.join(_HERE, "weights", "model_4ch_noaug.pth")
WEIGHTS_AUG = os.path.join(_HERE, "weights", "model_4ch_aug.pth")
OUTPUT = os.path.join(_HERE, "output.png")

MODEL_SIZE = 224
SIGMA = 10.0
PROMPT_SEED = 85775926
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_heatmap(center, size=(MODEL_SIZE, MODEL_SIZE), sigma=SIGMA):
    h, w = size
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)


def dice_score(pred, gt):
    pred, gt = pred > 0.5, gt > 0.5
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(2.0 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-8))


def sample_prompt_from_mask(mask, seed=PROMPT_SEED):
    # The prompt simulates a clinician's click: a random point inside the
    # wound region, fixed by a seed so the demo is reproducible.
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return MODEL_SIZE // 2, MODEL_SIZE // 2
    np.random.seed(seed)
    i = np.random.randint(len(ys))
    px = min(max(int(xs[i] * MODEL_SIZE / w), 0), MODEL_SIZE - 1)
    py = min(max(int(ys[i] * MODEL_SIZE / h), 0), MODEL_SIZE - 1)
    return px, py


def load_model(weights_path):
    model = smp.DeepLabV3Plus("resnet50", in_channels=4, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def main():
    img_bgr = cv2.imread(IMAGE)
    img_vis = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (MODEL_SIZE, MODEL_SIZE))
    gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (MODEL_SIZE, MODEL_SIZE))
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    norm = (gray_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    img_t = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    mask = np.load(MASK)
    mask224 = cv2.resize((mask > 0).astype(np.float32), (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_NEAREST)

    px, py = sample_prompt_from_mask(mask > 0)
    hmap_t = torch.tensor(create_heatmap((px, py)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        inp = torch.cat([img_t, hmap_t], dim=1)
        pred_noaug = torch.sigmoid(load_model(WEIGHTS_NOAUG)(inp)).cpu().numpy()[0, 0]
        pred_aug = torch.sigmoid(load_model(WEIGHTS_AUG)(inp)).cpu().numpy()[0, 0]

    dsc_noaug = dice_score(pred_noaug, mask224)
    dsc_aug = dice_score(pred_aug, mask224)
    print(f"DSC  no-aug = {dsc_noaug:.3f}  |  aug = {dsc_aug:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.92, bottom=0.01)
    panels = [
        ("Original", None, None, None),
        ("Ground Truth", mask224, "Greens", None),
        ("4ch (No Aug)", pred_noaug > 0.5, "Purples", (dsc_noaug, "purple")),
        ("4ch (Augmented)", pred_aug > 0.5, "Blues", (dsc_aug, "darkblue")),
    ]
    for ax, (title, overlay, cmap, score) in zip(axes, panels):
        ax.imshow(img_vis)
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, alpha=0.6)
        if score is not None:
            ax.scatter(px, py, c="yellow", marker="*", s=300, edgecolors="black")
            dsc, color = score
            ax.text(0.95, 0.05, f"DSC: {dsc:.3f}", transform=ax.transAxes,
                    color=color, fontsize=26, fontweight="bold", ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.1))
        ax.set_title(title, fontsize=18)
        ax.axis("off")

    plt.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
