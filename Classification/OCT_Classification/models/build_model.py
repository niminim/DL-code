import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import timm

import numpy as np

def get_model(model_name, in_channels, num_classes, device):
    if model_name == 'efficientnet':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1', progress=True)
        model.features[0][0] = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3),stride=(2,2), padding=(1,1),bias=False)
        model.classifier[1] = nn.Linear(1280,num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(1024,num_classes)

    return model.to(device)
