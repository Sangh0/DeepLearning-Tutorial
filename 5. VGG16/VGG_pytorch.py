import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


# Direct Implementation
class VGG16(nn.Module):

    def __init__(self, in_dim=3, hidden_dim=64, out_dim=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(),
            nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


# Load Pretrained Model in PyTorch
class VGG16(nn.Module):
    
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x