import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# ===========================
# Checkpoints and saving models
# ===========================

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, filename="checkpoint.pth"):
    """Save model and optimizer state dict to a file"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# ===========================
# Logging metrics
# ===========================

def log_metrics(epoch, loss, accuracy, log_dir, mode='train'):
    """Log metrics to TensorBoard"""
    writer = SummaryWriter(log_dir=log_dir)
    
    # Add metrics to TensorBoard
    writer.add_scalar(f'{mode}/Loss', loss, epoch)
    writer.add_scalar(f'{mode}/Accuracy', accuracy, epoch)
    
    writer.close()
    print(f"Metrics logged for {mode} at epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

# ===========================
# Visualization functions
# ===========================

def visualize_predictions(images, labels, preds, task="classification", num_images=5):
    """Visualize predictions vs. ground truth for classification/segmentation"""
    if task == "classification":
        visualize_classification(images, labels, preds, num_images)
    elif task == "segmentation":
        visualize_segmentation(images, labels, preds, num_images)
    else:
        raise ValueError(f"Unknown task: {task}")

def visualize_classification(images, labels, preds, num_images=5):
    """Visualize a few classification images with their labels and predictions"""
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert from CHW to HWC format
        ax.set_title(f"GT: {labels[i].item()}, Pred: {preds[i].item()}")
        ax.axis('off')
    plt.show()

def visualize_segmentation(images, labels, preds, num_images=5):
    """Visualize a few segmentation images with their ground truth and predictions"""
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Convert from CHW to HWC format
        ax.imshow(preds[i].cpu().numpy(), alpha=0.5, cmap='jet')  # Overlay prediction
        ax.set_title(f"GT: {labels[i].sum().item() > 0}, Pred: {preds[i].sum().item() > 0}")
        ax.axis('off')
    plt.show()

def save_predictions(images, labels, preds, output_dir, prefix="segmentation", num_images=5):
    """Save images of predictions, labels, and inputs"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # CHW to HWC
        label = labels[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        
        # Save original image
        plt.imsave(os.path.join(output_dir, f"{prefix}_image_{i}.png"), image)
        
        # Save label and prediction
        plt.imsave(os.path.join(output_dir, f"{prefix}_label_{i}.png"), label, cmap='gray')
        plt.imsave(os.path.join(output_dir, f"{prefix}_pred_{i}.png"), pred, cmap='jet')

    print(f"Predictions saved to {output_dir}")
