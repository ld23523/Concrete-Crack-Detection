import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random

from dataset import CrackDataset
from models.resnet import ResNet18Classifier
from transforms import get_classification_transforms
from config import DEVICE, IMAGE_SIZE, NUM_CLASSES

# SDNET paths
TRAIN_DIR = "data/processed_sdnet/train"
VAL_DIR = "data/processed_sdnet/val"
TEST_DIR = "data/processed_sdnet/test"

BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001


def validate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    print("Training on SDNET → Testing on SDNET")

    train_transform = get_classification_transforms(IMAGE_SIZE, train=True)
    test_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    # Load FULL dataset first
    train_data = CrackDataset(TRAIN_DIR, task="classification", image_size=IMAGE_SIZE, transform=train_transform)
    val_data = CrackDataset(VAL_DIR, task="classification", image_size=IMAGE_SIZE, transform=test_transform)
    test_data = CrackDataset(TEST_DIR, task="classification", image_size=IMAGE_SIZE, transform=test_transform)

    # Apply 25% subset ONLY to training
    fraction = 0.25
    subset_size = int(len(train_data) * fraction)

    indices = list(range(len(train_data)))
    random.shuffle(indices)

    train_data = Subset(train_data, indices[:subset_size])

    print(f"Train size (25%): {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18Classifier(pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1} finished")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    test_loss, test_acc = validate(model, test_loader, criterion)

    print("\nFinal SDNET Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()