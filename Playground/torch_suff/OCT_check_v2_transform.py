import os
import numpy as np

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

cnfg = {'bs_train': 16,
        'bs_test': 16,
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# new v2
train_transform_v2 = v2.Compose(
    [
        v2.Resize((224,224)),
        v2.RandomRotation((-20,20)),
        transforms.ToTensor(),
        # v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

test_transform_v2 = v2.Compose(
    [
        v2.Resize((224,224)),
        transforms.ToTensor(),
        # v2.ToDtype(torch.float32, scale=True),
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

############### Simple Train Code
epoch_times = []
for epoch in range(10):
    print(f"epoch: {epoch}")

    # Record the start event for the epoch
    starter_epoch, ender_epoch = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter_epoch.record()

    for i, (imgs, labels) in enumerate(test_loader, 0):
        inputs, labels = imgs.to(device), labels.to(device)  # inputs.dtype - torch.float32, and labels.dtype - torch.int64

    # Record the end event for the epoch
    ender_epoch.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    epoch_time = starter_epoch.elapsed_time(ender_epoch)/1000.0
    epoch_times.append(epoch_time)

mean_value = np.mean(epoch_times)
std_dev = np.std(epoch_times, ddof=1)  # ddof=1 to get the sample standard deviation

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_dev}")


# Regular Transform (Resize + To Tensor + Normalize)
# 16 - 1.0688 +- 0.025
# 32 - 1.085 +- 0.0268
# 64 - 1.126 +- 0.038


# V2 Transform (Resize + To Tensor + Normalize)
# 16 - 1.043 +- 0.0093
# 32 - 1.081 +- 0.027
# 64 - 1.111 +- 0.0261

#######################################


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

for epoch in range(2):
    print(f'Epoch: {epoch+1}')
    train_loss = 0.0
    correct, total = 0, 0

    print(f'lr: {scheduler.get_lr()[0]}')
    lr = scheduler.optimizer.param_groups[0]['lr']
    for i, (imgs, labels) in enumerate(test_loader, 0):
        inputs, labels = imgs.to(device), labels.to(device) # inputs.dtype - torch.float32, and labels.dtype - torch.int64

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

###############################




########################### Check transform directly on random data
import torch
from torchvision import transforms
from PIL import Image
from time import time

# Example batch of images (on CPU for transformations)
batch_images = torch.randn(512, 3, 448, 448, dtype=torch.float32)

# Define your transformations
train_transform = transforms.Compose([ # all - 0.000603
    transforms.Resize((224, 224)),  # (h, w) # without - 0.000334
    transforms.RandomRotation((-20, 20)), # without - 0.000186
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # without - 0.000549
])


# train_transform = v2.Compose(
#     [
#         v2.Resize((224,224)),
#         v2.RandomRotation((-20,20)),
#         transforms.ToTensor(),
#         # v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]
# )

times = []
# Apply transformations to each image in the batch
transformed_batch = []
for img_tensor in batch_images:
    # Convert tensor to PIL Image
    pil_img = transforms.ToPILImage()(img_tensor)

    # Apply the composed transformations
    time_start = time()
    transformed_img = train_transform(pil_img)
    time_end = time()
    times.append(time_end-time_start)

    # Append transformed image to the list
    transformed_batch.append(transformed_img)

# Stack the transformed images back into a batch
transformed_batch = torch.stack(transformed_batch)

# Verify the transformed batch
print(transformed_batch.shape)  # Should output (32, 3, 224, 224)
print(np.mean(times))
#################################