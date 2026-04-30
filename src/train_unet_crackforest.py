# U-Net Training for CrackForest Dataset
import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms.functional as TF


# Settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "data/raw/crackforest/Images"
MASK_DIR = "data/raw/crackforest/Masks"

OUTPUT_DIR = "outputs"
MODEL_DIR = "models/saved_models"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# Dataset

class CrackForestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        # Example:
        # image: 001.jpg
        # mask: 001_label.PNG
        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + "_label.PNG"

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))

        # Simple augmentation
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Convert mask to binary: 0 or 1
        mask = (mask > 0.5).float()

        return image, mask

# U-Net Model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.final_conv(u1)


# Metrics

def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
    return dice.item()


def iou_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()

# Training / Evaluation

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks)
        total_iou += iou_score(outputs, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    return avg_loss, avg_dice, avg_iou


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    return avg_loss, avg_dice, avg_iou

# Save Graphs / Predictions

def save_training_graph(history):
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net CrackForest Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "unet_crackforest_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_dice"], label="Train Dice")
    plt.plot(df["epoch"], df["val_dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("U-Net CrackForest Dice Score")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "unet_crackforest_dice.png"))
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_iou"], label="Train IoU")
    plt.plot(df["epoch"], df["val_iou"], label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("U-Net CrackForest IoU Score")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "unet_crackforest_iou.png"))
    plt.close()


def save_prediction_examples(model, loader, num_examples=5):
    model.eval()

    images, masks = next(iter(loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    for i in range(min(num_examples, images.size(0))):
        image = images[i].permute(1, 2, 0).numpy()
        mask = masks[i][0].numpy()
        pred = preds[i][0].numpy()

        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"unet_prediction_example_{i+1}.png"))
        plt.close()

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    full_dataset = CrackForestDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        augment=True
    )

    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    print(f"Total images: {total_size}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_dice = -1

    history = []

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou
        })

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "unet_crackforest_history.csv"),
        index=False
    )

    save_training_graph(history)

    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "unet_crackforest_best.pth")
    )

    test_loss, test_dice, test_iou = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n==============================")
    print("Final U-Net Test Results")
    print("==============================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    print(f"Test IoU Score: {test_iou:.4f}")

    pd.DataFrame([{
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou
    }]).to_csv(
        os.path.join(OUTPUT_DIR, "unet_crackforest_test_results.csv"),
        index=False
    )

    save_prediction_examples(model, test_loader, num_examples=5)

    print("\nDone.")
    print("Saved model: models/saved_models/unet_crackforest_best.pth")
    print("Saved graphs and results in outputs/")


if __name__ == "__main__":
    main()