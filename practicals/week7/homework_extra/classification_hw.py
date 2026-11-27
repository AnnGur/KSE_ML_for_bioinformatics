"""Homework step 1 (blood-cell classification): baseline CNN vs one architecture change vs one augmentation.

Same data, split, model code and optimiser as the in-class part of Practical_07 (stratified 80/10/10 split,
seed 221026, 128 px, batch 32, Adam lr 1e-3). Differences: every condition is trained for the same 10 epochs
and repeated over 3 seeds, so that the comparison can be separated from run-to-run noise.

Usage: python classification_hw.py <path to bloodcells_dataset> <output json>
"""
import json
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

DATA_ROOT, OUT_JSON = sys.argv[1], sys.argv[2]
RANDOM_STATE, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR = 221026, 128, 32, 10, 1e-3
SEEDS = [221026, 221027, 221028]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

# ---------------------------------------------------------------- data (same split as the notebook)
dataset = datasets.ImageFolder(DATA_ROOT)
class_names = dataset.classes
targets = np.asarray(dataset.targets)
indices = np.arange(len(dataset))
train_idx, tmp_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=targets)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=RANDOM_STATE, stratify=targets[tmp_idx])
assert (len(train_idx), len(val_idx), len(test_idx)) == (13673, 1709, 1710), (len(train_idx), len(val_idx), len(test_idx))

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
original_sizes, original_modes, arrays = Counter(), Counter(), []
t0 = time.time()
for path, _ in dataset.samples:
    with Image.open(path) as image:
        original_sizes[f"{image.width}x{image.height}"] += 1
        original_modes[image.mode] += 1
        arrays.append(np.asarray(resize(image.convert("RGB")), dtype=np.uint8))
X = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()      # N x 3 x 128 x 128, uint8
Y = torch.from_numpy(targets)
print(f"loaded {len(X)} images in {time.time() - t0:.0f}s", flush=True)

pixels = X.float() / 255.0
image_properties = {
    "n_images": len(X),
    "original_sizes": dict(original_sizes.most_common()),
    "original_modes": dict(original_modes),
    "resized_shape": list(X.shape[1:]),
    "dtype_after_ToTensor": "float32 in [0, 1]",
    "pixel_min": float(pixels.min()), "pixel_max": float(pixels.max()),
    "channel_mean": [round(float(v), 4) for v in pixels.mean(dim=(0, 2, 3))],
    "channel_std": [round(float(v), 4) for v in pixels.std(dim=(0, 2, 3))],
    "class_counts_total": {c: int((targets == i).sum()) for i, c in enumerate(class_names)},
    "class_counts_train": {c: int((targets[train_idx] == i).sum()) for i, c in enumerate(class_names)},
    "class_counts_val": {c: int((targets[val_idx] == i).sum()) for i, c in enumerate(class_names)},
    "class_counts_test": {c: int((targets[test_idx] == i).sum()) for i, c in enumerate(class_names)},
}
del pixels


# ---------------------------------------------------------------- model (copied from the notebook)
class ConfigurableCNN(nn.Module):
    def __init__(self, in_channels, n_classes, image_size=128, depth=2, base_channels=16, batch_norm=False,
                 dropout=0.0, global_average_pooling=False):
        super().__init__()
        blocks, channels = [], []
        current_channels = in_channels
        for level in range(depth):
            output_channels = base_channels * (2 ** level)
            blocks.append(nn.Conv2d(current_channels, output_channels, kernel_size=3, padding=1))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(output_channels))
            blocks.extend([nn.ReLU(), nn.MaxPool2d(2)])
            channels.append(output_channels)
            current_channels = output_channels
        self.features = nn.Sequential(*blocks)
        if global_average_pooling:
            classifier_input = channels[-1]
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            reduced = image_size // (2 ** depth)
            classifier_input = channels[-1] * reduced * reduced
            self.pool = nn.Identity()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(classifier_input, n_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


CONDITIONS = {
    "baseline": dict(model=dict(), augment=False),
    "batch normalization": dict(model=dict(batch_norm=True), augment=False),
    "augmentation (flip + rotation)": dict(model=dict(), augment=True),
}


def augment_batch(batch):
    out = []
    for image in batch:
        if torch.rand(1).item() < 0.5:                                  # RandomHorizontalFlip(p=0.5)
            image = TF.hflip(image)
        angle = float(torch.empty(1).uniform_(-10, 10))                 # RandomRotation(10)
        out.append(TF.rotate(image, angle))
    return torch.stack(out)


@torch.no_grad()
def evaluate(model, index):
    model.eval()
    losses, preds = 0.0, []
    for start in range(0, len(index), 256):
        batch = index[start:start + 256]
        logits = model(X[batch].float().div(255).to(DEVICE))
        losses += nn.functional.cross_entropy(logits, Y[batch].to(DEVICE), reduction="sum").item()
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    true = Y[index].numpy()
    return {"loss": losses / len(index), "accuracy": float((preds == true).mean()),
            "macro_f1": float(f1_score(true, preds, average="macro"))}


def run(condition, seed):
    cfg = CONDITIONS[condition]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConfigurableCNN(3, len(class_names), IMAGE_SIZE, **cfg["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        total, correct = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x = X[batch].float().div(255)
            if cfg["augment"]:
                x = augment_batch(x)
            x, y = x.to(DEVICE), Y[batch].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            correct += int((logits.argmax(1) == y).sum())
        val = evaluate(model, val_idx)
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "train_acc": correct / len(order),
                        "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]})
    test = evaluate(model, test_idx)                                    # final-epoch model, scored once
    return {"condition": condition, "seed": seed, "n_parameters": sum(p.numel() for p in model.parameters()),
            "history": history, "final_val": evaluate(model, val_idx), "final_test": test}


runs = []
for seed in SEEDS:
    for condition in CONDITIONS:
        t0 = time.time()
        result = run(condition, seed)
        runs.append(result)
        print(f"{condition:32s} seed {seed} | val acc {result['final_val']['accuracy']:.3f} | "
              f"test acc {result['final_test']['accuracy']:.3f} | {time.time() - t0:.0f}s", flush=True)
        json.dump({"config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "image_size": IMAGE_SIZE,
                              "seeds": SEEDS, "split_seed": RANDOM_STATE, "device": str(DEVICE),
                              "torch": torch.__version__},
                   "image_properties": image_properties, "classes": class_names, "runs": runs},
                  open(OUT_JSON, "w"), indent=1)
print("done", flush=True)
