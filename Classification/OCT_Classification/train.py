import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim

from Classification.OCT_Classification.models.build_model import get_model
from Classification.OCT_Classification.test_utils import *


# print(os.getcwd()) # get current working directory
# sys.path.append('/home/nim/venv/DL-code/Classification/OCT_Classification')
# os.chdir('/home/nim/venv/DL-code/Classification')

input_size = 224
model_name = 'mixnet' # efficientnet, mobilenetv3, mixnet


###### Train from folder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = get_model(model_name, in_channels=3, num_classes=4, device=device) # in_channels = 1 from CSV, 3 - from Folder

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

from time import time
t0 = time()
for epoch in range(10):  # loop over the dataset multiple times
    print(f'epoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'lr: {scheduler.get_lr()[0]}')
    lr = scheduler.optimizer.param_groups[0]['lr']
    for i, (img, label) in enumerate(train_loader, 0):
        inputs, labels = img.to(device), label.to(device) # inputs.dtype and labels.dtype - torch.int64

        optimizer.zero_grad()
        outputs = model(inputs) # inputs.dtype and inputs.dtype - torch.float32
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'running loss: {running_loss/len(train_loader):.2f}')
    print(f'train acc: {100*(correct/total):.2f}%')
    print('*****')
    scheduler.step()
    running_loss = 0.0
print('Finished Training')
t1 = time()
print(f'time: {t1-t0}')


probs_and_labels, val_preds, true_labels = evaluate(model, val_dataset, val_loader, device)
cm, cr = get_metrics(val_preds, true_labels, val_dataset)
####


############# ###### Train from CSV
from sklearn.preprocessing import LabelEncoder

model = get_model(model_name, in_channels=1, num_classes=4, device=device) # in_channels = 1 from CSV, 3 - from Folder

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

for epoch in range(40):  # loop over the dataset multiple times
    print(f'epoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'lr: {scheduler.get_lr()[0]}')
    lr = scheduler.optimizer.param_groups[0]['lr']
    for i, (img, label, img_data) in enumerate(train_loader, 0):
        label_encoder = LabelEncoder()
        label_encoder.fit(label)
        numerical_labels = torch.Tensor(label_encoder.transform(label)).long()
        inputs, labels = img.to(device), numerical_labels.to(device) # inputs.dtype and labels.dtype - torch.int64

        optimizer.zero_grad()
        outputs = model(inputs) # inputs.dtype and inputs.dtype - torch.float32
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'running loss: {running_loss/len(train_loader):.2f}')
    print(f'train acc: {100*(correct/total):.2f}%')
    scheduler.step()
    running_loss = 0.0
print('Finished Training')


correct, total = 0, 0
all_preds = torch.empty(0,).to(device)
all_probs = torch.empty(0, 4).to(device) # when creating dataset from CSV there's no .classes

model.eval()
with torch.no_grad():
    for data in val_loader:
        for i, (img, label, img_data) in enumerate(val_loader, 0):
            label_encoder = LabelEncoder()
            label_encoder.fit(label)
            numerical_labels = torch.Tensor(label_encoder.transform(label)).long()
            inputs, labels = img.to(device), numerical_labels.to(device)  # inputs.dtype and labels.dtype - torch.int64

            outputs = model(inputs)
            val_probs = torch.nn.functional.softmax(outputs, dim=1)[:,:4]

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


cm = confusion_matrix(true_labels.cpu(), all_preds.cpu())
cr = classification_report(true_labels.cpu(), all_preds.cpu(), target_names=list(val_dataset.class_to_idx.keys()))

print('cm')
print(cm)
print('cr')
print(cr)