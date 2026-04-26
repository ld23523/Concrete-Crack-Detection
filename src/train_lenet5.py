import os
import copy
import time
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# Basic settings
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = "outputs"
MODEL_DIR = "models/saved_models"

IMAGE_SIZE = 64
DATA_FRACTION = 0.25   # use only 25% of each split

BASELINE_EPOCHS = 6
TUNING_EPOCHS = 5


# ==============================
# LeNet-5 Model
# ==============================

class LeNet5(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # For image size 64x64, output becomes 16 x 13 x 13
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==============================
# Data loading
# ==============================

def make_subset(dataset, fraction=0.25):
    """
    Creates a stratified subset so we only use part of the data,
    while keeping crack/no_crack balance.
    """
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    subset_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=42,
        stratify=labels
    )

    return Subset(dataset, subset_indices)


def get_dataloaders(batch_size=32, data_fraction=0.25):
    """
    Loads train, validation, and test image folders.
    Uses only 25% of each split.
    """

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_dataset = make_subset(train_dataset, data_fraction)
    val_dataset = make_subset(val_dataset, data_fraction)
    test_dataset = make_subset(test_dataset, data_fraction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = ["crack", "no_crack"]

    return train_loader, val_loader, test_loader, class_names


# ==============================
# Evaluation helper
# ==============================

def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


# ==============================
# Training function
# ==============================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Trains a model and keeps the version with the best validation F1.
    """
    best_weights = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    history = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate_model(
            model,
            val_loader,
            criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision_macro": val_precision,
            "val_recall_macro": val_recall,
            "val_macro_f1": val_f1
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return model, history, best_val_f1


# ==============================
# Graph functions
# ==============================

def save_training_graph(history, output_path):
    """
    Saves F1 score graph for train vs validation.
    """
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_macro_f1"], label="Train Macro F1")
    plt.plot(df["epoch"], df["val_macro_f1"], label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("LeNet-5 Training vs Validation Macro F1")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_loss_graph(history, output_path):
    """
    Saves loss graph for train vs validation.
    """
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LeNet-5 Training vs Validation Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix_graph(cm, class_names, output_path):
    """
    Saves confusion matrix as an image.
    """
    plt.figure()
    plt.imshow(cm)
    plt.title("LeNet-5 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.savefig(output_path)
    plt.close()


# ==============================
# Baseline model
# ==============================

def run_baseline():
    print("\n==============================")
    print("Running LeNet-5 Baseline")
    print("==============================")
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=32,
        data_fraction=DATA_FRACTION
    )

    model = LeNet5(dropout_rate=0.3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, history, best_val_f1 = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=BASELINE_EPOCHS
    )

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_baseline_history.csv"),
        index=False
    )

    save_training_graph(
        history,
        os.path.join(OUTPUT_DIR, "lenet5_baseline_f1_graph.png")
    )

    save_loss_graph(
        history,
        os.path.join(OUTPUT_DIR, "lenet5_baseline_loss_graph.png")
    )

    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "lenet5_baseline.pth")
    )

    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred = evaluate_model(
        model,
        test_loader,
        criterion
    )

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix_graph(
        cm,
        class_names,
        os.path.join(OUTPUT_DIR, "lenet5_baseline_confusion_matrix.png")
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_baseline_classification_report.csv")
    )

    print("\nBaseline Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision Macro: {test_precision:.4f}")
    print(f"Test Recall Macro: {test_recall:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return model


# ==============================
# Hyperparameter tuning
# ==============================

def run_hyperparameter_tuning():
    print("\n==============================")
    print("Running LeNet-5 Hyperparameter Tuning")
    print("==============================")

    param_grid = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [16, 32, 64],
    "dropout_rate": [0.2, 0.3, 0.4]
}

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    tuning_results = []

    best_model = None
    best_params = None
    best_val_f1 = -1

    for combo in combinations:
        params = dict(zip(keys, combo))

        print("\nTesting parameters:")
        print(params)

        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            batch_size=params["batch_size"],
            data_fraction=DATA_FRACTION
        )

        model = LeNet5(dropout_rate=params["dropout_rate"]).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

        start_time = time.time()

        model, history, val_f1 = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=TUNING_EPOCHS
        )

        runtime = time.time() - start_time

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate_model(
            model,
            val_loader,
            criterion
        )

        tuning_results.append({
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "dropout_rate": params["dropout_rate"],
            "optimizer": "adam",
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision_macro": val_precision,
            "val_recall_macro": val_recall,
            "val_macro_f1": val_f1,
            "runtime_seconds": runtime
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
            best_model = copy.deepcopy(model)

    results_df = pd.DataFrame(tuning_results)
    results_df = results_df.sort_values(by="val_macro_f1", ascending=False)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_hyperparameter_tuning_results.csv"),
        index=False
    )

    torch.save(
        best_model.state_dict(),
        os.path.join(MODEL_DIR, "lenet5_best_tuned.pth")
    )

    print("\nBest Hyperparameters:")
    print(best_params)
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=best_params["batch_size"],
        data_fraction=DATA_FRACTION
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred = evaluate_model(
        best_model,
        test_loader,
        criterion
    )

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix_graph(
        cm,
        class_names,
        os.path.join(OUTPUT_DIR, "lenet5_tuned_confusion_matrix.png")
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_tuned_classification_report.csv")
    )

    print("\nBest Tuned Model Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision Macro: {test_precision:.4f}")
    print(f"Test Recall Macro: {test_recall:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return best_model, best_params


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    #run_baseline()
    run_hyperparameter_tuning()

    print("\nDone. Results saved in outputs/ and models/saved_models/")