import os

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

import torchvision
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import v2



base_split_folder = '/home/nim/Downloads/Data/OCT2017'
train_data_path = os.path.join(base_split_folder, 'train')
test_data_path = os.path.join(base_split_folder, 'test')

cnfg = {'bs_train': 512,
        'bs_test': 512,
        'num_classes': 4}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

### transforms
# regular
train_transform = transforms.Compose([
    transforms.Resize((224,224)), # (h,w)
    transforms.RandomRotation((-20,20)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)), # (h,w)
    transforms.ToTensor(),
])

# new v2
train_transform_v2 = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomRotation((-20,20)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

test_transform_v2 = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


### datasets & dataloaders
train_dataset = ImageFolder(train_data_path, transform=train_transform)
test_dataset = ImageFolder(test_data_path, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cnfg['bs_train'],
                                           shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cnfg['bs_test'],
                                         shuffle=False, num_workers=2)


# model
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1') # in_channels = 1 from CSV, 3 - from Folder
model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2),
                                 padding=(1, 1), bias=False)
model.classifier[3] = nn.Linear(1024, cnfg['num_classes'])
model.to(device)


# optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

for epoch in range(10):
    print(f'Epoch: {epoch+1}')
    train_loss = 0.0
    correct, total = 0, 0

    print(f'lr: {scheduler.get_lr()[0]}')
    lr = scheduler.optimizer.param_groups[0]['lr']
    for i, (imgs, labels) in enumerate(train_loader, 0):
        inputs, labels = imgs.to(device), labels.to(device) # inputs.dtype and labels.dtype - torch.int64

        optimizer.zero_grad()
        outputs = model(inputs) # inputs.dtype and inputs.dtype - torch.float32
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * (correct / total)
    print(f'running loss: {train_loss:.3f}')
    print(f'train acc: {train_acc:.3f}%')
    scheduler.step()
    print('*****')




