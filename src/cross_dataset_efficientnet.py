"""
cross_dataset_efficientnet.py

Cross-dataset generalization experiments for EfficientNet-B3.
Experiment 4 from the project proposal:
  - Train on Kaggle dataset, test on SDNET2018
  - Train on SDNET2018, test on Kaggle dataset
"""

import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from dataset import CrackDataset
from models.efficientnet import EfficientNetB3Classifier
from transforms import get_classification_transforms
from config import (
    LOG_DIR,
    FIGURES_DIR,
    NUM_EPOCHS,
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
    BATCH_SIZE,
)

# ======================
# Paths
# ======================
# Kaggle dataset — uses PROCESSED_DATA_DIR from config (same as resnet)
KAGGLE_DIR = "data/processed/concrete_crack_75_10_15"

# SDNET2018 dataset — matches path used in cross_dataset_resnet.py
SDNET_TEST_DIR = "data/processed_sdnet/test"

OUTPUT_DIR = "outputs/efficientnet_cross_dataset"
CSV_PATH = os.path.join(OUTPUT_DIR, "efficientnet_cross_dataset_results.csv")
DETAILS_PATH = os.path.join(OUTPUT_DIR, "efficientnet_cross_dataset_output.txt")

CLASS_NAMES = ["crack", "no_crack"]


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


def write_log(text):
    print(text)
    with open(DETAILS_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, cm, report


def save_confusion_matrix(cm, experiment_name):
    safe_name = experiment_name.replace(" ", "_").replace("/", "_").replace("->", "to")
    fig_path = os.path.join(OUTPUT_DIR, "figures", f"{safe_name}_confusion_matrix.png")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix\n{experiment_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def train_model(train_dir, experiment_name, learning_rate=0.0001, batch_size=16):
    """Train EfficientNet-B3 on a given dataset and return trained model."""
    write_log(f"\nTraining on: {train_dir}")

    set_random_seed(SEED)

    train_transform = get_classification_transforms(IMAGE_SIZE, train=True)
    val_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    train_data = CrackDataset(
        os.path.join(train_dir, "train"),
        task="classification",
        image_size=IMAGE_SIZE,
        transform=train_transform
    )
    val_data = CrackDataset(
        os.path.join(train_dir, "val"),
        task="classification",
        image_size=IMAGE_SIZE,
        transform=val_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = EfficientNetB3Classifier(pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    write_log(f"Train size: {len(train_data)} | Val size: {len(val_data)}")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                write_log(
                    f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        write_log(
            f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

    # Save training curve
    safe_name = experiment_name.replace(" ", "_").replace("/", "_").replace("->", "to")
    curve_path = os.path.join(OUTPUT_DIR, "figures", f"{safe_name}_training_curve.png")
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.title(f"Training Curve: {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    write_log(f"  Saved training curve: {curve_path}")

    return model, criterion


def test_on_dataset(model, criterion, test_dir, experiment_name, batch_size=16):
    """Test a trained model on a dataset. test_dir should point directly to the folder of images."""
    write_log(f"\nTesting on: {test_dir}")

    val_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    test_data = CrackDataset(
        test_dir,
        task="classification",
        image_size=IMAGE_SIZE,
        transform=val_transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    write_log(f"Test size: {len(test_data)}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, test_report = evaluate(
        model, test_loader, criterion
    )

    write_log(f"\nResults for: {experiment_name}")
    write_log(f"  Test Loss:      {test_loss:.4f}")
    write_log(f"  Test Accuracy:  {test_acc:.4f}")
    write_log(f"  Test Precision: {test_prec:.4f}")
    write_log(f"  Test Recall:    {test_rec:.4f}")
    write_log(f"  Test F1:        {test_f1:.4f}")
    write_log(f"\nClassification Report:\n{test_report}")
    write_log(f"Confusion Matrix:\n{test_cm}")

    cm_path = save_confusion_matrix(test_cm, experiment_name)
    write_log(f"Saved confusion matrix: {cm_path}")

    return {
        "experiment": experiment_name,
        "model": "EfficientNet-B3",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "confusion_matrix_path": cm_path,
    }


def save_results_csv(results):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    write_log(f"\nCSV results saved to: {CSV_PATH}")


def plot_cross_dataset_comparison(results):
    """Bar chart comparing cross-dataset generalization performance."""
    names = [r["experiment"] for r in results]
    accuracies = [r["test_accuracy"] for r in results]
    f1s = [r["test_f1"] for r in results]

    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], accuracies, width, label="Accuracy")
    ax.bar([i + width/2 for i in x], f1s, width, label="F1 Score")

    ax.set_ylabel("Score")
    ax.set_title("EfficientNet-B3: Cross-Dataset Generalization")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "figures", "cross_dataset_comparison.png")
    plt.savefig(path)
    plt.close()
    write_log(f"Saved cross-dataset comparison chart: {path}")


def main():
    make_dirs()

    if os.path.exists(DETAILS_PATH):
        os.remove(DETAILS_PATH)

    write_log("EfficientNet-B3 Cross-Dataset Generalization Experiments")
    write_log(f"Using device: {DEVICE}")
    write_log(f"NUM_EPOCHS: {NUM_EPOCHS}")

    results = []

    # =================================================
    # Train on Kaggle, test on both Kaggle and SDNET2018
    # (matches cross_dataset_resnet.py pattern)
    # =================================================
    write_log("\n" + "=" * 80)
    write_log("Training EfficientNet-B3 on Kaggle dataset...")
    write_log("=" * 80)

    kaggle_train_dir = os.path.join(KAGGLE_DIR, "train")
    kaggle_val_dir   = os.path.join(KAGGLE_DIR, "val")
    kaggle_test_dir  = os.path.join(KAGGLE_DIR, "test")

    model, criterion = train_model(
        train_dir=KAGGLE_DIR,
        experiment_name="EfficientNet-B3 Kaggle Training",
        learning_rate=0.0001,
        batch_size=16
    )

    # Test on Kaggle (in-domain)
    result_kaggle = test_on_dataset(
        model, criterion,
        test_dir=kaggle_test_dir,
        experiment_name="Kaggle Test (in-domain)"
    )
    results.append(result_kaggle)

    # Test on SDNET (cross-domain) — path matches resnet version
    if os.path.exists(SDNET_TEST_DIR):
        result_sdnet = test_on_dataset(
            model, criterion,
            test_dir=SDNET_TEST_DIR,
            experiment_name="SDNET Test (cross-domain)"
        )
        results.append(result_sdnet)
    else:
        write_log(f"\nSDNET folder not found! Expected: {SDNET_TEST_DIR}")

    save_results_csv(results)
    plot_cross_dataset_comparison(results)

    write_log("\n" + "=" * 80)
    write_log("CROSS-DATASET EXPERIMENTS COMPLETE")
    write_log("=" * 80)
    write_log("\nSummary:")
    for r in results:
        write_log(
            f"  {r['experiment']:45s} | "
            f"Acc={r['test_accuracy']:.4f} | "
            f"F1={r['test_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
