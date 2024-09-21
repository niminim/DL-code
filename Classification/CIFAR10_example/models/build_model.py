import torch.nn as nn
from torchvision import models
import timm

def get_model(config, device):

    model_name = config['model_name'].lower()
    num_classes = config['num_classes']

    if model_name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1), bias=False)
        model.classifier[3] = nn.Linear(1024, num_classes)
    elif model_name == 'mixnet_s':
        model = timm.create_model('mixnet_s', pretrained=True)
        model.conv_stem = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2),
                                    padding=(1, 1), bias=False)
        model.classifier = nn.Linear(1536, num_classes)
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
        model.conv_stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                    padding=(1, 1), bias=False)
        model.classifier = nn.Linear(1280, num_classes)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='IMAGENET1K_V1', progress=True)
        model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='IMAGENET1K_V1', progress=True)
        model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(1408, num_classes)
    elif model_name == 'cnn':
        model = CNN(num_classes)

    print(f'Created {model_name} model')
    return model.to(device)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x