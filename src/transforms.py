# src/transforms.py

import torchvision.transforms as T
from PIL import Image
import random
import torch

# ===========================
# Classification Transforms
# ===========================
def get_classification_transforms(image_size=227, train=True):
    """
    Returns torchvision transforms for classification.
    
    Args:
        image_size (int): Target image size
        train (bool): Whether to return training or validation transforms
    """
    if train:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation / test transforms
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    return transform


# ===========================
# Segmentation Transforms
# ===========================
class SegmentationTransform:
    """
    Custom transform class for segmentation datasets.
    Applies the same random transforms to both image and mask.
    """
    def __init__(self, image_size=480, train=True):
        self.image_size = image_size
        self.train = train

    def __call__(self, image, mask):
        # Resize
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))

        # Random horizontal flip
        if self.train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if self.train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Color jitter for image only
        if self.train:
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)
            image = color_jitter(image)

        # Convert to tensor
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)  # mask will be 1xHxW

        # Normalize image
        image = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(image)

        return image, mask