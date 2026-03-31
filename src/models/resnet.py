import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 model for concrete crack classification.
    
    Args:
        pretrained (bool): If True, use ImageNet pretrained weights
        num_classes (int): Number of output classes
    """
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer with a new one
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ===========================
# Helper function
# ===========================
def get_resnet18_model(pretrained=True):
    """
    Returns ResNet-18 model moved to DEVICE
    """
    model = ResNet18Classifier(pretrained=pretrained)
    return model.to(DEVICE)