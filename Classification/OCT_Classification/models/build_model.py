import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import timm

import numpy as np


def get_model(model_name, num_classes, device):
    if model_name == 'efficientnet':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1', progress=True)
        model.classifier[1] = nn.Linear(1280,num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(1024,num_classes)

    return model.to(device)
