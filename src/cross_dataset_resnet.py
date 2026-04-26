import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import CrackDataset
from models.resnet import ResNet18Classifier
from transforms import get_classification_transforms
from config import (
    PROCESSED_DATA_DIR,
    NUM_EPOCHS,
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
)


# Dataset paths


# Kaggle dataset (used for training)
KAGGLE_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
KAGGLE_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
KAGGLE_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

# SDNET dataset (used for cross-dataset testing)
SDNET_TEST_DIR = "data/processed_sdnet/test"

# Output folder
OUTPUT_DIR = "outputs/cross_dataset_resnet"
CSV_PATH = os.path.join(OUTPUT_DIR, "cross_dataset_results.csv")

CLASS_NAMES = ["crack", "no_crack"]



# Set random seed

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Create folders
def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# Evaluate model
def evaluate(model, loader, criterion, dataset_name):

    model.eval()

    total_loss = 0
    y_true = []
    y_pred = []

    # Disable gradient calculation
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predicted class
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Print results
    print("\n==============================")
    print(f"Results on {dataset_name}")
    print("==============================")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "dataset": dataset_name,
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }



# Train ResNet on Kaggle

def train_resnet():

    set_random_seed(SEED)

    # Data transforms
    train_transform = get_classification_transforms(IMAGE_SIZE, train=True)
    val_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    # Load datasets
    # Load full training dataset
    train_data = CrackDataset(
        KAGGLE_TRAIN_DIR,
        task="classification",
        image_size=IMAGE_SIZE,
        transform=train_transform
    )

    # Use only 25% of training data
    train_fraction = 0.25
    subset_size = int(len(train_data) * train_fraction)

    indices = list(range(len(train_data)))
    random.shuffle(indices)

    train_data = Subset(train_data, indices[:subset_size])

    print(f"Using {subset_size} / {len(indices)} training samples (25%)")    
    val_data = CrackDataset(KAGGLE_VAL_DIR, task="classification", image_size=IMAGE_SIZE, transform=val_transform)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    # Load pretrained ResNet18
    model = ResNet18Classifier(pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    return model, criterion



# Main function

def main():

    make_dirs()

    # Train model on Kaggle
    model, criterion = train_resnet()

    # Use same transform for testing
    test_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    results = []

  
    # Test on Kaggle (normal test)

    kaggle_test = CrackDataset(KAGGLE_TEST_DIR, task="classification", image_size=IMAGE_SIZE, transform=test_transform)
    kaggle_loader = DataLoader(kaggle_test, batch_size=16, shuffle=False)

    results.append(evaluate(model, kaggle_loader, criterion, "Kaggle Test"))

    
    # Cross-dataset test (SDNET)
   
    if os.path.exists(SDNET_TEST_DIR):

        sdnet_test = CrackDataset(SDNET_TEST_DIR, task="classification", image_size=IMAGE_SIZE, transform=test_transform)
        sdnet_loader = DataLoader(sdnet_test, batch_size=16, shuffle=False)

        results.append(evaluate(model, sdnet_loader, criterion, "SDNET Test"))

    else:
        print("\nSDNET folder not found!")
        print("Expected: data/processed_sdnet/test/")

 
    # Save results
   
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to:", CSV_PATH)


if __name__ == "__main__":
    main()