import torch.nn as nn
from torch import optim
from Classification.OCT_Classification.test_utils import *
from Classification.OCT_Classification.models.build_model import get_model

from sklearn.preprocessing import LabelEncoder


def get_model_loss_optim(config, device):
    model = get_model(config, device=device) # in_channels = 1 from CSV, 3 - from Folder
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
    return model, criterion, optimizer, scheduler


def train_epoch(epoch, model, optimizer, scheduler, criterion, train_loader, device, logger):
    # train one epoch
    print(f'Epoch: {epoch}')
    train_loss = 0.0
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
        train_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * (correct / total)
    print(f'running loss: {train_loss:.2f}')
    print(f'train acc: {train_acc:.2f}%')
    logger.info(f'Epoch {epoch} completed. Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}%')
    scheduler.step()

    return train_loss, train_acc, probs, predicted

def train_epoch_csv(epoch, model, optimizer, scheduler, criterion, train_loader, device, logger):

    print(f'Epoch: {epoch}')
    train_loss = 0.0
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
        train_loss += loss.item()

        scores, predicted = torch.max(outputs.data, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * (correct / total)
    print(f'running loss: {train_loss:.2f}')
    print(f'train acc: {train_acc:.2f}%')
    logger.info(f'Epoch {epoch} completed. Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}%')
    scheduler.step()

    return train_loss, train_acc,  probs, predicted
