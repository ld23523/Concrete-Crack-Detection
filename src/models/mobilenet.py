# src/models/mobilenet_model.py

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE

class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 model for concrete crack classification.
    
    Args:
        pretrained (bool): If True, use ImageNet pretrained weights
        num_classes (int): Number of output classes
    """
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super(MobileNetV2Classifier, self).__init__()
        
        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace the classifier with a new linear layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


# ===========================
# Helper function
# ===========================
def get_mobilenetv2_model(pretrained=True):
    """
    Returns MobileNetV2 model moved to DEVICE
    """
    model = MobileNetV2Classifier(pretrained=pretrained)
    return model.to(DEVICE)