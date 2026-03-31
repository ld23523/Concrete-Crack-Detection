import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CrackDataset
from src.models import (
    resnet,
    efficientnet,
    mobilenet,
    unet,
)
from src.utils import log_metrics, visualize_predictions, save_predictions
from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    DEVICE,
    MODEL_NAME,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
    TASK,
)

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
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = correct_preds / total_preds
    precision, recall, f1 = calculate_classification_metrics(all_labels, all_preds)
    return val_loss / len(val_loader), accuracy, precision, recall, f1

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
    preds = (preds > 0.5).float()
    intersection = (preds & labels).sum()
    union = (preds | labels).sum()
    return intersection / union if union != 0 else 0.0

def calculate_dice(preds, labels):
    """Calculate Dice coefficient for segmentation"""
    preds = (preds > 0.5).float()
    intersection = (preds & labels).sum()
    total = preds.sum() + labels.sum()
    return 2 * intersection / total if total != 0 else 0.0

def main():
    set_random_seed(SEED)

    # Load the dataset
    val_data = CrackDataset(PROCESSED_DATA_DIR, task="classification", image_size=IMAGE_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load model based on config
    if MODEL_NAME == "mobilenet_v2":
        model = mobilenet.MobileNetV2(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "resnet18":
        model = resnet.ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "efficientnet_b3":
        model = efficientnet.EfficientNetB3(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "unet":
        model = unet.get_unet_model(in_channels=3, out_channels=1).to(DEVICE)
    else:
        raise ValueError("Invalid model name")

    # Load the checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer to load the checkpoint
    model, optimizer, epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Choose the appropriate criterion
    if MODEL_NAME == "unet":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    if "classification" in TASK:
        val_loss, accuracy, precision, recall, f1 = evaluate_classification(model, val_loader, criterion)
        log_metrics(epoch, val_loss, accuracy, LOG_DIR, mode="validation")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    elif "segmentation" in TASK:
        val_loss, iou, dice = evaluate_segmentation(model, val_loader, criterion)
        log_metrics(epoch, val_loss, iou, LOG_DIR, mode="validation")
        print(f"Validation Loss: {val_loss:.4f}, IoU: {iou:.4f}, Dice Coefficient: {dice:.4f}")
        visualize_predictions(val_loader.dataset[:5], task="segmentation")  # Visualize first 5 images

if __name__ == "__main__":
    main()