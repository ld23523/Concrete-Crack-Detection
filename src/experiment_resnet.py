import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from dataset import CrackDataset
from models.resnet import ResNet18Classifier
from transforms import get_classification_transforms
from config import (
    PROCESSED_DATA_DIR,
    LOG_DIR,
    FIGURES_DIR,
    NUM_EPOCHS,
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
)

# ======================
# Experiment settings
# ======================
OUTPUT_DIR = "outputs/resnet_full_experiment"
CSV_PATH = os.path.join(OUTPUT_DIR, "resnet18_full_results.csv")
DETAILS_PATH = os.path.join(OUTPUT_DIR, "resnet18_detailed_output.txt")

CLASS_NAMES = ["crack", "no_crack"]


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
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

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1, cm, report


def save_confusion_matrix(cm, experiment_name):
    fig_path = os.path.join(
        OUTPUT_DIR,
        "figures",
        f"{experiment_name.replace(' ', '_').replace('/', '_')}_confusion_matrix.png"
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix: {experiment_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def save_training_curve(history, experiment_name):
    safe_name = experiment_name.replace(" ", "_").replace("/", "_")
    fig_path = os.path.join(OUTPUT_DIR, "figures", f"{safe_name}_training_curve.png")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title(f"Loss Curve: {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def save_accuracy_curve(history, experiment_name):
    safe_name = experiment_name.replace(" ", "_").replace("/", "_")
    fig_path = os.path.join(OUTPUT_DIR, "figures", f"{safe_name}_accuracy_curve.png")

    epochs = list(range(1, len(history["val_accuracy"]) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.plot(epochs, history["val_f1"], marker="o", label="Validation F1")
    plt.title(f"Accuracy/F1 Curve: {experiment_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def train_one_experiment(
    experiment_name,
    learning_rate,
    batch_size,
    train_fraction=1.0,
    augment=False
):
    write_log("\n" + "=" * 80)
    write_log(f"Starting experiment: {experiment_name}")
    write_log(f"Learning Rate: {learning_rate}")
    write_log(f"Batch Size: {batch_size}")
    write_log(f"Train Fraction: {train_fraction}")
    write_log(f"Augmentation: {augment}")
    write_log(f"Device: {DEVICE}")
    write_log("=" * 80)

    set_random_seed(SEED)

    train_transform = get_classification_transforms(IMAGE_SIZE, train=augment)
    val_transform = get_classification_transforms(IMAGE_SIZE, train=False)

    train_data = CrackDataset(
        os.path.join(PROCESSED_DATA_DIR, "train"),
        task="classification",
        image_size=IMAGE_SIZE,
        transform=train_transform
    )

    val_data = CrackDataset(
        os.path.join(PROCESSED_DATA_DIR, "val"),
        task="classification",
        image_size=IMAGE_SIZE,
        transform=val_transform
    )

    test_data = CrackDataset(
        os.path.join(PROCESSED_DATA_DIR, "test"),
        task="classification",
        image_size=IMAGE_SIZE,
        transform=val_transform
    )

    original_train_size = len(train_data)

    if train_fraction < 1.0:
        subset_size = int(original_train_size * train_fraction)
        indices = list(range(original_train_size))
        random.shuffle(indices)
        train_data = Subset(train_data, indices[:subset_size])
    else:
        subset_size = original_train_size

    write_log(f"Original train size: {original_train_size}")
    write_log(f"Used train size: {subset_size}")
    write_log(f"Validation size: {len(val_data)}")
    write_log(f"Test size: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = ResNet18Classifier(
        pretrained=True,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = -1
    best_result = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                write_log(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm, val_report = evaluate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)

        write_log("-" * 60)
        write_log(f"Epoch {epoch + 1}/{NUM_EPOCHS} finished")
        write_log(f"Train Loss: {avg_train_loss:.4f}")
        write_log(f"Val Loss: {val_loss:.4f}")
        write_log(f"Val Accuracy: {val_acc:.4f}")
        write_log(f"Val Precision: {val_prec:.4f}")
        write_log(f"Val Recall: {val_rec:.4f}")
        write_log(f"Val F1: {val_f1:.4f}")
        write_log("Validation Classification Report:")
        write_log(val_report)
        write_log(f"Validation Confusion Matrix:\n{val_cm}")
        write_log("-" * 60)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_result = {
                "experiment": experiment_name,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "train_fraction": train_fraction,
                "augmentation": augment,
                "used_train_size": subset_size,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }

    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, test_report = evaluate(
        model,
        test_loader,
        criterion
    )

    best_result["test_loss"] = test_loss
    best_result["test_accuracy"] = test_acc
    best_result["test_precision"] = test_prec
    best_result["test_recall"] = test_rec
    best_result["test_f1"] = test_f1

    cm_path = save_confusion_matrix(test_cm, experiment_name)
    loss_curve_path = save_training_curve(history, experiment_name)
    acc_curve_path = save_accuracy_curve(history, experiment_name)

    best_result["confusion_matrix_path"] = cm_path
    best_result["loss_curve_path"] = loss_curve_path
    best_result["accuracy_curve_path"] = acc_curve_path

    write_log("\nFinal Test Results:")
    write_log(f"Test Loss: {test_loss:.4f}")
    write_log(f"Test Accuracy: {test_acc:.4f}")
    write_log(f"Test Precision: {test_prec:.4f}")
    write_log(f"Test Recall: {test_rec:.4f}")
    write_log(f"Test F1: {test_f1:.4f}")
    write_log("Test Classification Report:")
    write_log(test_report)
    write_log(f"Test Confusion Matrix:\n{test_cm}")
    write_log(f"Saved confusion matrix: {cm_path}")
    write_log(f"Saved loss curve: {loss_curve_path}")
    write_log(f"Saved accuracy curve: {acc_curve_path}")

    return best_result


def save_results_csv(results):
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    write_log(f"\nCSV results saved to: {CSV_PATH}")


def plot_summary_graphs(results):
    figures_dir = os.path.join(OUTPUT_DIR, "figures")

    # Graph 1: Validation F1 by experiment
    names = [r["experiment"] for r in results]
    f1_scores = [r["val_f1"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(names, f1_scores)
    plt.xticks(rotation=90)
    plt.ylabel("Validation F1")
    plt.title("Validation F1 by Experiment")
    plt.tight_layout()
    path = os.path.join(figures_dir, "summary_val_f1_by_experiment.png")
    plt.savefig(path)
    plt.close()
    write_log(f"Saved graph: {path}")

    # Graph 2: Test accuracy by experiment
    test_acc = [r["test_accuracy"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(names, test_acc)
    plt.xticks(rotation=90)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Experiment")
    plt.tight_layout()
    path = os.path.join(figures_dir, "summary_test_accuracy_by_experiment.png")
    plt.savefig(path)
    plt.close()
    write_log(f"Saved graph: {path}")

    # Graph 3: Data size experiment only
    data_results = [r for r in results if "Data Size" in r["experiment"]]

    if data_results:
        fractions = [r["train_fraction"] * 100 for r in data_results]
        scores = [r["test_f1"] for r in data_results]

        plt.figure(figsize=(7, 5))
        plt.plot(fractions, scores, marker="o")
        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Test F1")
        plt.title("Effect of Training Data Size on ResNet18")
        plt.tight_layout()
        path = os.path.join(figures_dir, "data_size_vs_test_f1.png")
        plt.savefig(path)
        plt.close()
        write_log(f"Saved graph: {path}")


def main():
    make_dirs()

    if os.path.exists(DETAILS_PATH):
        os.remove(DETAILS_PATH)

    write_log("ResNet18 Full Experiment Started")
    write_log(f"NUM_EPOCHS from config.py: {NUM_EPOCHS}")
    write_log(f"Using device: {DEVICE}")

    results = []

    # =================================================
    # 1. Data size experiment
    # =================================================
    for fraction in [0.25]:
        result = train_one_experiment(
            experiment_name=f"Data Size {int(fraction * 100)}%",
            learning_rate=0.0001,
            batch_size=16,
            train_fraction=fraction,
            augment=False
        )
        results.append(result)

    # =================================================
    # 2. Hyperparameter tuning / grid search
    # =================================================
    learning_rates = [0.001, 0.0001, 0.00001]
    batch_sizes = [8, 16, 32]

    for lr in learning_rates:
        for bs in batch_sizes:
            result = train_one_experiment(
                experiment_name=f"Grid Search LR={lr} BS={bs}",
                learning_rate=lr,
                batch_size=bs,
                train_fraction=0.25,
                augment=False
            )
            results.append(result)

    # =================================================
    # 3. Data augmentation experiment
    # =================================================
    for aug in [False, True]:
        result = train_one_experiment(
            experiment_name=f"Augmentation {aug}",
            learning_rate=0.0001,
            batch_size=16,
            train_fraction=0.25,
            augment=aug
        )
        results.append(result)

    save_results_csv(results)
    plot_summary_graphs(results)

    write_log("\n" + "=" * 80)
    write_log("ALL EXPERIMENTS COMPLETE")
    write_log("=" * 80)

    write_log("\nFinal Summary:")
    for r in results:
        write_log(
            f"{r['experiment']} | "
            f"LR={r['learning_rate']} | "
            f"BS={r['batch_size']} | "
            f"Data={r['train_fraction']} | "
            f"Aug={r['augmentation']} | "
            f"Val Acc={r['val_accuracy']:.4f} | "
            f"Val F1={r['val_f1']:.4f} | "
            f"Test Acc={r['test_accuracy']:.4f} | "
            f"Test F1={r['test_f1']:.4f}"
        )

    write_log(f"\nCSV file: {CSV_PATH}")
    write_log(f"Detailed output file: {DETAILS_PATH}")
    write_log(f"Graphs folder: {os.path.join(OUTPUT_DIR, 'figures')}")


if __name__ == "__main__":
    main()