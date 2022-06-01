import torch.nn as nn
from torchvision.models import convnext_tiny


def convnext_t_encoder(pretrained=True):
    model = convnext_tiny(pretrained=pretrained)
    model.classifier = nn.Identity()
    return model