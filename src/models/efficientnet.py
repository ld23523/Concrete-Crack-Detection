import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from config import NUM_CLASSES, DEVICE

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 model for concrete crack classification.
    
    Args:
        pretrained (bool): If True, use ImageNet pretrained weights
        num_classes (int): Number of output classes
    """
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super(EfficientNetB3Classifier, self).__init__()
        
        # Load pretrained EfficientNet-B3
        self.model = EfficientNet.from_pretrained('efficientnet-b3' if pretrained else None)
        
        # Replace the classifier with a new linear layer
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
    
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