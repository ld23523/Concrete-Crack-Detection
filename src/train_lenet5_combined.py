#this script trains a LeNet-5 model on a combined dataset of the original concrete crack dataset and the SDNET dataset. It uses 25% of the training data from both datasets for training, and evaluates on a combined validation set as well as the SDNET test set. The script saves training history, graphs, model weights, and evaluation results in the outputs/ and models/saved_models/ directories.
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
BATCH_SIZE = 32
DATA_FRACTION = 0.25
EPOCHS = 8
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

OUTPUT_DIR = "outputs"
MODEL_DIR = "models/saved_models"

# Original dataset
ORIG_TRAIN_DIR = "data/processed/train"
ORIG_VAL_DIR = "data/processed/val"

# SDNET dataset
SDNET_TRAIN_DIR = "data/processed_sdnet/train"
SDNET_VAL_DIR = "data/processed_sdnet/val"
SDNET_TEST_DIR = "data/processed_sdnet/test"


# Model

class LeNet5(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

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


# Helpers


def make_subset(dataset, fraction=0.25):
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    subset_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=42,
        stratify=labels
    )

    return Subset(dataset, subset_indices)


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
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

    orig_train = datasets.ImageFolder(ORIG_TRAIN_DIR, transform=train_transform)
    sdnet_train = datasets.ImageFolder(SDNET_TRAIN_DIR, transform=train_transform)

    orig_val = datasets.ImageFolder(ORIG_VAL_DIR, transform=eval_transform)
    sdnet_val = datasets.ImageFolder(SDNET_VAL_DIR, transform=eval_transform)

    sdnet_test = datasets.ImageFolder(SDNET_TEST_DIR, transform=eval_transform)

    # Use only 25% from each dataset
    orig_train_subset = make_subset(orig_train, DATA_FRACTION)
    sdnet_train_subset = make_subset(sdnet_train, DATA_FRACTION)

    orig_val_subset = make_subset(orig_val, DATA_FRACTION)
    sdnet_val_subset = make_subset(sdnet_val, DATA_FRACTION)

    sdnet_test_subset = make_subset(sdnet_test, DATA_FRACTION)

    # Combine original + SDNET for training and validation
    combined_train = ConcatDataset([orig_train_subset, sdnet_train_subset])
    combined_val = ConcatDataset([orig_val_subset, sdnet_val_subset])

    train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(sdnet_test_subset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = sdnet_test.classes

    print("Original train 25%:", len(orig_train_subset))
    print("SDNET train 25%:", len(sdnet_train_subset))
    print("Combined train size:", len(combined_train))
    print("Combined val size:", len(combined_val))
    print("SDNET test 25%:", len(sdnet_test_subset))
    print("Classes:", class_names)

    return train_loader, val_loader, test_loader, class_names


def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, precision, recall, f1, all_labels, all_preds


def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_weights = copy.deepcopy(model.state_dict())
    best_val_f1 = -1

    history = []

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
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

            train_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
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
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return model, history


def save_graph(history, output_path):
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_macro_f1"], label="Train Macro F1")
    plt.plot(df["epoch"], df["val_macro_f1"], label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Combined Dataset LeNet-5 F1")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(cm, class_names, output_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Combined Training → SDNET Test Confusion Matrix")
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


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    model = LeNet5(dropout_rate=DROPOUT_RATE).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer
    )

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_combined_25_history.csv"),
        index=False
    )

    save_graph(
        history,
        os.path.join(OUTPUT_DIR, "lenet5_combined_25_f1_graph.png")
    )

    # Save model
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "lenet5_combined_25.pth")
    )

    # Final test on SDNET test set only
    test_loss, test_acc, test_precision, test_recall, test_f1, y_true, y_pred = evaluate_model(
        model,
        test_loader,
        criterion
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\n==============================")
    print("Combined Training → SDNET Test Results")
    print("==============================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision Macro: {test_precision:.4f}")
    print(f"Test Recall Macro: {test_recall:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    pd.DataFrame([{
        "training_data": "25% original train + 25% SDNET train",
        "test_data": "25% SDNET test",
        "loss": test_loss,
        "accuracy": test_acc,
        "precision_macro": test_precision,
        "recall_macro": test_recall,
        "macro_f1": test_f1
    }]).to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_combined_25_test_results.csv"),
        index=False
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_combined_25_classification_report.csv")
    )

    save_confusion_matrix(
        cm,
        class_names,
        os.path.join(OUTPUT_DIR, "lenet5_combined_25_confusion_matrix.png")
    )

    print("\nDone. Results saved in outputs/ and model saved in models/saved_models/")


if __name__ == "__main__":
    main()