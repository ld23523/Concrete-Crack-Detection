# src/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

# ===========================
# Classification Dataset
# ===========================
class CrackClassificationDataset(Dataset):
    """
    PyTorch Dataset for concrete crack image classification.
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of integer labels (0 = crack, 1 = no crack)
        transform (callable, optional): Optional transform to apply to images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# ===========================
# Segmentation Dataset
# ===========================
class CrackSegmentationDataset(Dataset):
    """
    PyTorch Dataset for concrete crack image segmentation.
    
    Args:
        image_paths (list): List of image file paths
        mask_paths (list): List of mask file paths (1 = crack, 0 = background)
        transform (callable, optional): Optional transform to apply to image and mask
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths), "Images and masks must have the same length"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Ensure mask is single-channel
        mask = mask.convert("L")

        # Apply transform if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask