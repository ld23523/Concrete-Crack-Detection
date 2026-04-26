import os
import torch
import torch.nn as nn
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
BATCH_SIZE = 32
DATA_FRACTION = 0.25

MODEL_PATH = "models/saved_models/lenet5_best_tuned.pth"

# Use SDNET test set as external dataset
EXTERNAL_DATA_DIR = "data/processed_sdnet/test"

OUTPUT_DIR = "outputs"


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


def save_confusion_matrix_graph(cm, class_names, output_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("LeNet-5 Cross-Dataset Confusion Matrix")
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"External dataset path: {EXTERNAL_DATA_DIR}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = datasets.ImageFolder(
        EXTERNAL_DATA_DIR,
        transform=transform
    )

    external_dataset = make_subset(full_dataset, DATA_FRACTION)

    external_loader = DataLoader(
        external_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = full_dataset.classes

    print("External dataset classes:", class_names)
    print(f"Full external dataset size: {len(full_dataset)}")
    print(f"Using subset size: {len(external_dataset)}")

    model = LeNet5(dropout_rate=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in external_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(external_dataset)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================")
    print("LeNet-5 Cross-Dataset Results")
    print("==============================")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision Macro: {precision:.4f}")
    print(f"Recall Macro: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_cross_dataset_classification_report.csv")
    )

    pd.DataFrame([{
        "dataset": "processed_sdnet_test_25_percent",
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1
    }]).to_csv(
        os.path.join(OUTPUT_DIR, "lenet5_cross_dataset_results.csv"),
        index=False
    )

    save_confusion_matrix_graph(
        cm,
        class_names,
        os.path.join(OUTPUT_DIR, "lenet5_cross_dataset_confusion_matrix.png")
    )

    print("\nDone. Results saved in outputs/")


if __name__ == "__main__":
    main()