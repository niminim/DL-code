import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models
import torch.optim as optim

import numpy as np

input_size = 192
model_name = 'efficientnet' # mobilenet, efficientnet

sys.path.append('/home/nim/venv/DL-code/Classification/OCT_Classification')

print(os.getcwd()) # get current working directory


os.chdir('/home/nim/venv/DL-code/Classification')
print(os.getcwd())

from OCT_Classification.Models.build_model import get_model

from Models.build_model import get_model
from DL_code.Classification.OCT_Classification.Models.build_model import get_model

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24,
                                          shuffle=False, num_workers=2)

######
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = get_model(model_name, num_classes, device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

for epoch in range(1):  # loop over the dataset multiple times
    print(f'epoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'lr: {scheduler.get_lr()[0]}')
    lr = scheduler.optimizer.param_groups[0]['lr']
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device) # inputs.dtype and labels.dtype - torch.int64

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs) # inputs.dtype and inputs.dtype - torch.float32
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'running loss: {running_loss:.2f}')
    print(f'train acc: {(correct/total):.3f}%')
    scheduler.step()
    running_loss = 0.0
print('Finished Training')




correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
all_preds = torch.empty(0,).to(device)
all_probs = torch.empty(0, len(val_dataset.classes)).to(device)

model.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        inputs, labels = data[0].to(device), data[1].to(device) # inputs.dtype - torch.float32, labels.dtype - torch.int64
        # calculate outputs by running images through the network
        outputs = model(inputs)
        val_probs = torch.nn.functional.softmax(outputs, dim=1)[:,:len(val_dataset.classes)]

        # the class with the highest energy is what we choose as prediction
        val_scores, val_predicted = torch.max(outputs.data, 1)
        all_preds = torch.cat((all_preds, val_predicted),axis=0)
        all_probs = torch.cat((all_probs, val_probs),axis=0)
        total += labels.size(0)
        correct += (val_predicted == labels).sum().item()

print(f'Accuracy of the model on the val-set images: {100 * (correct/total):.2f} %')
print('val_dataset.targets: ',val_dataset.targets)
print('all_preds: ',all_preds)

true_labels = torch.Tensor(val_dataset.targets).reshape(len(val_dataset),1)
final = torch.cat((all_probs, true_labels.to(device)), dim=1)
final = torch.round(final, decimals=3)