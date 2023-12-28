import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

def get_model(config, device):
    model_name = config['model']['architecture'].lower()
    num_classes = config['model']['num_classes']
    in_channels = config['model']['in_channels']

    if model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1', progress=True)
        model.features[0][0] = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3),stride=(2,2), padding=(1,1),bias=False)
        model.classifier[1] = nn.Linear(1280,num_classes)
    elif model_name == 'efficientnet_b0':
            model = timm.create_model('efficientnet_b0', pretrained=True)
            model.conv_stem = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3),stride=(2,2), padding=(1,1),bias=False)
            model.classifier = nn.Linear(1280,num_classes)
    elif model_name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.features[0][0] = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3),stride=(2,2), padding=(1,1),bias=False)
        model.classifier[3] = nn.Linear(1024,num_classes)
    elif model_name == 'mixnet':
        model = timm.create_model('mixnet_s', pretrained=True)
        model.conv_stem = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3),stride=(2,2), padding=(1,1),bias=False)
        model.classifier = nn.Linear(1536,num_classes)

    print(f'Created {model_name} model')
    return model.to(device)