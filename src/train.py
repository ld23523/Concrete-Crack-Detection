import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CrackDataset  # Assuming you have a custom Dataset for both tasks
from models import (
    resnet,
    efficientnet,
    mobilenet,
    unet,
)
from utils import save_checkpoint, log_metrics
from transforms import get_classification_transforms, SegmentationTransform
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    MODEL_NAME,
    DATASET_NAME,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEED,
    TASK,
)

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_classification(model, train_loader, val_loader, criterion, optimizer):
    """Training loop for classification task"""
    model.train()
    for epoch in range(NUM_EPOCHS):
        # Train loop
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # Log metrics (or use TensorBoard)
        log_metrics(epoch, val_loss, val_accuracy, LOG_DIR, mode="validation", metric_name="Accuracy")

        # Save the model checkpoint
        if epoch % 5 == 0:  # Save every 5 epochs (adjust as needed)
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, filename=f"{MODEL_NAME}_{DATASET_NAME}_checkpoint.pth")

def train_segmentation(model, train_loader, val_loader, criterion, optimizer):
    """Training loop for segmentation task"""
    model.train()
    for epoch in range(NUM_EPOCHS):
        # Train loop
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss, val_iou = validate_segmentation(model, val_loader, criterion)

        # Log metrics (IoU for segmentation)
        log_metrics(epoch, val_loss, val_iou, LOG_DIR, mode="validation", metric_name="IoU")

        # Save the model checkpoint
        if epoch % 5 == 0:  # Save every 5 epochs (adjust as needed)
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, filename=f"{MODEL_NAME}_{DATASET_NAME}_checkpoint.pth")

def validate(model, val_loader, criterion):
    """Validation loop for classification task"""
    model.eval()
    val_loss = 0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    accuracy = correct_preds / total_preds
    return val_loss / len(val_loader), accuracy

def validate_segmentation(model, val_loader, criterion):
    """Validation loop for segmentation task"""
    model.eval()
    val_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate IoU for segmentation
            iou = calculate_iou(outputs, masks)
            total_iou += iou

    return val_loss / len(val_loader), total_iou / len(val_loader)

def calculate_iou(preds, labels):
    """Calculate Intersection over Union (IoU) for segmentation"""
    preds = torch.sigmoid(preds) > 0.5
    preds = preds.float()
    intersection = (preds * labels).sum()
    union = (preds + labels).sum() - intersection
    return intersection / union if union != 0 else 0.0

def main():
    # Recompute PROCESSED_DATA_DIR
    if DATASET_NAME in ["concrete_crack", "sdnet"] and TASK == "classification":
        PROCESSED_DATA_DIR = "data/processed/concrete_crack_75_10_15"
    elif DATASET_NAME == "crackforest" and TASK == "segmentation":
        PROCESSED_DATA_DIR = "data/processed_segmentation"
    else:
        PROCESSED_DATA_DIR = "data/processed"
    
    set_random_seed(SEED)

    # Load dataset
    if TASK == "classification":
        train_transform = get_classification_transforms(IMAGE_SIZE, train=True)
        val_transform = get_classification_transforms(IMAGE_SIZE, train=False)
        train_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'train'), task="classification", image_size=IMAGE_SIZE, transform=train_transform)
        val_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'val'), task="classification", image_size=IMAGE_SIZE, transform=val_transform)
    elif TASK == "segmentation":
        train_transform = SegmentationTransform(IMAGE_SIZE, train=True)
        val_transform = SegmentationTransform(IMAGE_SIZE, train=False)
        train_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'train'), task="segmentation", image_size=IMAGE_SIZE, transform=train_transform)
        val_data = CrackDataset(os.path.join(PROCESSED_DATA_DIR, 'val'), task="segmentation", image_size=IMAGE_SIZE, transform=val_transform)
    else:
        raise ValueError("Invalid task. Should be 'classification' or 'segmentation'.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Select model based on config
    if MODEL_NAME == "mobilenet_v2":
        model = mobilenet.MobileNetV2Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "resnet18":
        model = resnet.ResNet18Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "efficientnet_b3":
        model = efficientnet.EfficientNetB3Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "unet":
        model = unet.get_unet_model(in_channels=3, out_channels=1).to(DEVICE)
    else:
        raise ValueError("Invalid model name")

    # Choose the criterion and optimizer
    criterion = nn.CrossEntropyLoss() if TASK == "classification" else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    if TASK == "classification":
        train_classification(model, train_loader, val_loader, criterion, optimizer)
    elif TASK == "segmentation":
        train_segmentation(model, train_loader, val_loader, criterion, optimizer)

    # Save final model weights
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_{DATASET_NAME}_final.pth'))
    print(f"Final model weights saved to {os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_{DATASET_NAME}_final.pth')}")

if __name__ == "__main__":
    main()