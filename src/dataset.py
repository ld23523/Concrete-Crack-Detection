import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CrackDataset(Dataset):
    """
    PyTorch Dataset for concrete crack detection, supporting both classification and segmentation tasks.

    Args:
        data_dir (str): Directory containing the image data.
        task (str): Either 'classification' or 'segmentation'.
        image_size (int): Size to resize the images to.
        transform (callable, optional): Optional transform to apply to images/masks.
    """
    def __init__(self, data_dir, task="classification", image_size=224, transform=None):
        self.data_dir = data_dir
        self.task = task
        self.image_size = image_size
        self.transform = transform
        
        # Initialize the dataset
        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        # Load data paths
        self._load_data()

    def _load_data(self):
        """Load image and mask paths for classification or segmentation."""
        if self.task == "classification":
            self._load_classification_data()
        elif self.task == "segmentation":
            self._load_segmentation_data()
        else:
            raise ValueError("Invalid task type. Use 'classification' or 'segmentation'.")

    def _load_classification_data(self):
        """Load image paths and labels for classification."""
        for label in ["crack", "no_crack"]:
            label_dir = os.path.join(self.data_dir, label)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(0 if label == "crack" else 1)

    def _load_segmentation_data(self):
        """Load image paths and corresponding mask paths for segmentation."""
        for img_name in os.listdir(self.data_dir):
            if img_name.endswith(".jpg") or img_name.endswith(".png"):
                self.image_paths.append(os.path.join(self.data_dir, img_name))
                mask_name = img_name.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
                self.mask_paths.append(os.path.join(self.data_dir, mask_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.task == "classification":
            label = self.labels[idx]
            return self._apply_transform(image, label)

        elif self.task == "segmentation":
            mask = Image.open(self.mask_paths[idx]).convert("L")  # Convert mask to grayscale
            return self._apply_transform(image, mask)

    def _apply_transform(self, image, target):
        """Apply the transform (if any) to image and target."""
        if self.transform:
            image = self.transform(image)
            target = self.transform(target) if isinstance(target, Image.Image) else target
        return image, target