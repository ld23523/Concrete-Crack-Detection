import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import CrackDataset
from models import (
    resnet,
    efficientnet,
    mobilenet,
    unet,
)
from transforms import get_classification_transforms, SegmentationTransform
from config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    DEVICE,
    DATASET_NAME,
    MODEL_NAME,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
    TASK,
)
from utils import log_metrics, save_results, save_predictions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from a checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def evaluate_classification(model, val_loader, criterion):
    """Evaluate classification model"""
    model.eval()
    val_loss = 0
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            if i % 50 == 0:
                print(f"Processed batch {i}")
    
    accuracy = correct_preds / total_preds
    precision, recall, f1 = calculate_classification_metrics(all_labels, all_preds)
    return val_loss / len(val_loader), accuracy, precision, recall, f1, all_labels, all_preds

def evaluate_segmentation(model, val_loader, criterion):
    """Evaluate segmentation model"""
    model.eval()
    val_loss = 0
    total_iou = 0
    total_dice = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate IoU and Dice coefficient for segmentation
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)
            
            total_iou += iou
            total_dice += dice
    
    return val_loss / len(val_loader), total_iou / len(val_loader), total_dice / len(val_loader)

def calculate_classification_metrics(labels, preds):
    """Calculate Precision, Recall, and F1-score for classification"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return precision, recall, f1

def calculate_iou(preds, labels):
    """Calculate Intersection over Union (IoU) for segmentation"""
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    intersection = (preds * labels).sum()
    union = (preds + labels).sum() - intersection
    return intersection / union if union != 0 else 0.0

def calculate_dice(preds, labels):
    """Calculate Dice coefficient for segmentation"""
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    intersection = (preds * labels).sum()
    total = preds.sum() + labels.sum()
    return 2 * intersection / total if total != 0 else 0.0

def main():
    set_random_seed(SEED)

    # Load the dataset
    if TASK == "classification":
        val_transform = get_classification_transforms(IMAGE_SIZE, train=False)
        val_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'val'), task="classification", image_size=IMAGE_SIZE, transform=val_transform)
    elif TASK == "segmentation":
        val_transform = SegmentationTransform(IMAGE_SIZE, train=False)
        val_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'val'), task="segmentation", image_size=IMAGE_SIZE, transform=val_transform)
    else:
        raise ValueError("Invalid task")
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load model based on config
    if MODEL_NAME == "mobilenet_v2":
        model = mobilenet.MobileNetV2Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "resnet18":
        model = resnet.ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "efficientnet_b3":
        model = efficientnet.EfficientNetB3(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "unet":
        model = unet.get_unet_model(in_channels=3, out_channels=1).to(DEVICE)
    else:
        raise ValueError("Invalid model name")

    # Load the checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_{DATASET_NAME}_checkpoint.pth")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer to load the checkpoint
    model, optimizer, epoch = load_checkpoint(model, optimizer, checkpoint_path)
    print(f"Loaded model from {checkpoint_path} at epoch {epoch}")
    
    # Choose the appropriate criterion
    if MODEL_NAME == "unet":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    if "classification" in TASK:
        val_loss, accuracy, precision, recall, f1, all_labels, all_preds = evaluate_classification(model, val_loader, criterion)
        epoch = 0  # Since we don't have epoch from state_dict
        log_metrics(epoch, val_loss, accuracy, LOG_DIR, mode="validation", metric_name="Accuracy")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save results to file
        results = {
            "Validation Loss": f"{val_loss:.4f}",
            "Accuracy": f"{accuracy:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1": f"{f1:.4f}"
        }
        results_path = os.path.join("outputs", "results.csv")
        save_results(results, results_path)
        
        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crack', 'Crack'], yticklabels=['No Crack', 'Crack'])
        plt.title(f'Confusion Matrix - {MODEL_NAME} on {DATASET_NAME}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{MODEL_NAME}_{DATASET_NAME}.png'))
        plt.close()
        
    elif "segmentation" in TASK:
        val_loss, iou, dice = evaluate_segmentation(model, val_loader, criterion)
        epoch = 0
        log_metrics(epoch, val_loss, iou, LOG_DIR, mode="validation", metric_name="IoU")
        print(f"Validation Loss: {val_loss:.4f}, IoU: {iou:.4f}, Dice Coefficient: {dice:.4f}")
        
        # Save results to file
        results = {
            "Validation Loss": f"{val_loss:.4f}",
            "IoU": f"{iou:.4f}",
            "Dice Coefficient": f"{dice:.4f}"
        }
        results_path = os.path.join("outputs", "results.csv")
        save_results(results, results_path)
        
        # Save predictions
        batch = next(iter(val_loader))
        images, masks = batch
        with torch.no_grad():
            preds = model(images.to(DEVICE)).cpu()
        save_predictions(images, masks, preds, os.path.join(FIGURES_DIR, f'predictions_{MODEL_NAME}_{DATASET_NAME}'), prefix="segmentation", num_images=5)

if __name__ == "__main__":
    main()