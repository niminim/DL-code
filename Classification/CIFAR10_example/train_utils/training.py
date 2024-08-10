import torch
import torch.nn as nn
import os
import json
from Classification.CIFAR10_example.train_utils.metrics import *


# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, config):

    top_k = config['top_k']
    num_classes = config['num_classes']
    num_epochs = config['num_epochs']
    models_dir = config['models_dir']
    history_file = config['history_file']

    best_val_acc = 0.0
    # num_epochs = 2
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print('LR: ', optimizer.param_groups[0]['lr'])
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        train_scores = torch.empty(0, num_classes)
        train_preds = torch.empty(0, 1)
        train_labels = torch.empty(0, 1)

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images) # (cuda, requires_grad)
            scores = torch.nn.functional.softmax(outputs, dim=1) # (cuda, requires_grad)
            _, preds = torch.max(outputs.data, 1) #  basically it's - descending scores, preds, but we don't want to overide the scores  (preds not on cuda)
            loss = criterion(outputs, labels)  # (cuda, requires_grad)                 

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            labels = torch.unsqueeze(labels.cpu(), dim=1) # reshape
            train_labels = torch.cat((train_labels, labels), dim=0)
            preds = torch.unsqueeze(preds, dim=1).cpu() # reshape
            train_preds = torch.cat((train_preds, preds), dim=0)
            train_scores = torch.cat((train_scores, scores.detach().cpu()), dim=0)
            correct += (preds == labels.cpu()).sum().item()

        # Calculate metrics
        train_metrics = compute_metrics(train_preds, train_scores, train_labels, running_loss, config)
        val_metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), config)


        # Print metrics at the end of each epoch
        print_metrics(phase='Train', metrics=train_metrics, top_k=config['top_k'])
        print_metrics(phase='Val', metrics=val_metrics, top_k=config['top_k'])

        # Update and save training history
        update_history(history_file, epoch, train_metrics, val_metrics)

        # Save the best model based on validation accuracy
        _, best_val_acc = save_best_model(model, epoch, val_metrics['accuracy'], best_val_acc, models_dir)
        
    return train_metrics, val_metrics

def evaluate_model(model, test_loader, config):
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), config)
    print("Test Performance:")
    print_metrics("test", test_metrics, config['top_k'])
    return test_metrics

def evaluate(model, loader, criterion, config):

    top_k = config['top_k']
    num_classes = config['num_classes']

    val_scores = torch.empty(0, config['num_classes'])
    val_preds = torch.empty(0, 1)
    val_labels = torch.empty(0, 1)

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            scores = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1) #  basically it's - scores, preds, but we don't want to overide the scores


            labels = torch.unsqueeze(labels.cpu(), dim=1) # reshape
            val_labels = torch.cat((val_labels, labels), dim=0)
            preds = torch.unsqueeze(preds, dim=1).cpu() # reshape
            val_preds = torch.cat((val_preds, preds), dim=0)
            val_scores = torch.cat((val_scores, scores.detach().cpu()), dim=0)


    metrics = compute_metrics(val_preds, val_scores, val_labels, val_loss, config)

    return metrics


def update_history(history_file, epoch, train_metrics, val_metrics):
    epoch_data = {
        'epoch': epoch + 1,
        'train': {k: round(v, 3) for k, v in train_metrics.items()},
        'val': {k: round(v, 3) for k, v in val_metrics.items()}
    }

    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history = json.load(file)

    history.append(epoch_data)

    with open(history_file, 'w') as file:
        json.dump(history, file, indent=4)




def save_best_model(model, epoch, val_acc, best_val_acc, save_dir):
    """
    Save the best model based on validation accuracy. Deletes the previous best model.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    epoch (int): The current epoch.
    val_accuracy (float): The validation accuracy for the current epoch.
    best_val_accuracy (float): The best validation accuracy observed so far.
    save_dir (str): Directory where the model will be saved.

    Returns:
    str: The path of the saved model.
    float: Updated best validation accuracy.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model if the current validation accuracy is the best seen so far
    if val_acc > best_val_acc:
        # Delete previous best model
        for filename in os.listdir(save_dir):
            if filename.startswith("best_model_"):
                os.remove(os.path.join(save_dir, filename))

        # Define the model filename with epoch and validation accuracy
        model_filename = f"best_model_epoch{epoch}_valacc_{val_acc:.3f}.pth"
        model_path = os.path.join(save_dir, model_filename)

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model: {model_filename}")

        # Update the best validation accuracy
        best_val_acc = val_acc

        return model_path, best_val_acc
    else:
        return None, best_val_acc
