import torch
import torch.nn as nn
import os
import json
from Classification.CIFAR10_example.utils.metrics import compute_metrics

# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes, top_k, history_file):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Initialize metrics for training
        train_metrics = initialize_metrics()

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            # Update metrics
            update_metrics(outputs, labels, num_classes, top_k, train_metrics)

        # Calculate average metrics for training
        train_metrics = calculate_average_metrics(train_metrics, running_loss)

        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, num_classes, top_k)

        # Print metrics at the end of each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print_metrics("train", train_metrics, top_k)
        print_metrics("validation", val_metrics, top_k)

        # Update and save training history
        update_history(history_file, epoch, train_metrics, val_metrics)

    return train_metrics, val_metrics

def evaluate_model(model, test_loader, num_classes, top_k):
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), num_classes, top_k)
    print("Test Performance:")
    print_metrics("test", test_metrics, top_k)
    return test_metrics

def evaluate(model, loader, criterion, num_classes, top_k):
    metrics = initialize_metrics()
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            update_metrics(outputs, labels, num_classes, top_k, metrics)

    metrics = calculate_average_metrics(metrics, total_loss)
    return metrics

def update_metrics(outputs, labels, num_classes, top_k, metrics):
    batch_metrics = compute_metrics(outputs, labels, num_classes, top_k)
    for key in batch_metrics:
        metrics[key] += round(batch_metrics[key], 3) * labels.size(0)
    metrics['total_samples'] += labels.size(0)

def initialize_metrics():
    return {
        'accuracy': 0.0,
        'class_accuracy': 0.0,
        'topk_acc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'auc': 0.0,
        'total_samples': 0
    }

def calculate_average_metrics(metrics, loss):
    avg_metrics = {key: round(metrics[key] / metrics['total_samples'], 3) for key in metrics if key != 'total_samples'}
    avg_metrics['loss'] = round(loss / metrics['total_samples'], 3)
    return avg_metrics

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

def print_metrics(phase, metrics, top_k):
    prefix = phase.capitalize()
    print(f'{prefix} - Loss: {metrics["loss"]:.3f}, '
          f'Accuracy: {metrics["accuracy"]:.3f}, '
          f'Class Accuracy: {metrics["class_accuracy"]:.3f}, '
          f'Top-{top_k} Accuracy: {metrics["topk_acc"]:.3f}, '
          f'F1 Score: {metrics["f1"]:.3f}, '
          f'Precision: {metrics["precision"]:.3f}, '
          f'Recall: {metrics["recall"]:.3f}, '
          f'AUC: {metrics["auc"]:.3f}')