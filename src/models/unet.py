import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

class UNet(nn.Module):
    """
    U-Net model for concrete crack segmentation.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images)
        out_channels (int): Number of output channels (1 for crack, 0 for background)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Load a pretrained ResNet-18 model as the encoder
        self.encoder = models.resnet18(pretrained=True)
        
        # Encoder layers for U-Net (based on ResNet)
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu,
            self.encoder.maxpool, self.encoder.layer1, self.encoder.layer2,
            self.encoder.layer3, self.encoder.layer4
        )
        
        # Decoder layers for U-Net
        self.upconv4 = self._upconv_block(512, 256)
        self.upconv3 = self._upconv_block(256, 128)
        self.upconv2 = self._upconv_block(128, 64)
        self.upconv1 = self._upconv_block(64, 32)
        
        # Final convolution layer to reduce to the desired output channels
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (downsampling)
        enc_out = self.encoder_layers(x)

        # Decoder (upsampling)
        x = self.upconv4(enc_out)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.upconv1(x)

        # Final output
        x = self.final_conv(x)
        
        return x

    def _upconv_block(self, in_channels, out_channels):
        """Upsampling block for the decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

# ===========================
# Helper function
# ===========================
def get_unet_model(in_channels=3, out_channels=1):
    """
    Returns U-Net model moved to DEVICE
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    return model.to(DEVICE)