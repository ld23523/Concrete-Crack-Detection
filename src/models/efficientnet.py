import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

# These may be imported from config.py in some contexts
try:
    from config import NUM_CLASSES, DEVICE
except ImportError:
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 model for concrete crack classification.
    Uses torchvision implementation (no external efficientnet_pytorch needed).

    Args:
        pretrained (bool): If True, use ImageNet pretrained weights
        num_classes (int): Number of output classes
    """
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super(EfficientNetB3Classifier, self).__init__()

        # Load EfficientNet-B3 from torchvision
        if pretrained:
            self.model = models.efficientnet_b3(
                weights=EfficientNet_B3_Weights.IMAGENET1K_V1
            )
        else:
            self.model = models.efficientnet_b3(weights=None)

        # Replace the classifier head for binary classification
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ===========================
# Helper function
# ===========================
def get_efficientnet_b3_model(pretrained=True):
    """
    Returns EfficientNet-B3 model moved to DEVICE
    """
    model = EfficientNetB3Classifier(pretrained=pretrained)
    return model.to(DEVICE)
