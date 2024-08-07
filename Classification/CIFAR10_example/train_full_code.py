import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system

# Configuration dictionary for parameters
config = {
    'batch_size': 100,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'data_dir': '/home/nim/data',
    'num_classes': 10,
    'val_split': 0.2,
    'top_k': 5,
    'history_file': '/home/nim/training_history.json'
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define a CNN model
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


# Function to define transformations
def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return transform_train, transform_test


# Function to load datasets
def load_data(data_dir, batch_size, val_split, transform_train, transform_test):
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform_train, download=True)

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform_test, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Function to initialize metrics dictionary
def initialize_metrics():
    metrics = {
        'accuracy': 0.0,
        'class_accuracy': 0.0,
        'topk_acc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'auc': 0.0,
        'total_samples': 0
    }
    return metrics


# Function to compute all relevant metrics
def compute_metrics(outputs, labels, num_classes, top_k=5):
    # Get predicted classes
    _, predicted = torch.max(outputs.data, 1)

    # Calculate Accuracy
    accuracy = accuracy_score(labels.cpu(), predicted.cpu())

    # Calculate Confusion Matrix
    confusion = compute_confusion_matrix(labels, predicted)

    # Calculate Per-Class Accuracy
    class_accuracy = confusion.diagonal() / confusion.sum(axis=1)
    class_accuracy = np.mean(class_accuracy)  # Average per-class accuracy

    # Calculate Top-k Accuracy
    topk_acc = compute_top_k_accuracy(outputs, labels, k=top_k)

    # Calculate F1 Score
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')

    # Calculate Precision
    precision = precision_score(labels.cpu(), predicted.cpu(), average='weighted', zero_division=0)

    # Calculate Recall
    recall = recall_score(labels.cpu(), predicted.cpu(), average='weighted', zero_division=0)

    # Calculate AUC for each class and average them
    auc = compute_auc(outputs, labels, num_classes)

    # Return metrics without confusion matrix
    metrics = {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'topk_acc': topk_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    return metrics, confusion


# Function to compute top-k accuracy
def compute_top_k_accuracy(outputs, labels, k=5):
    _, topk_preds = torch.topk(outputs, k, dim=1)
    topk_correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_accuracy = topk_correct.sum().float() / labels.size(0)
    return topk_accuracy.item()


# Function to compute confusion matrix
def compute_confusion_matrix(labels, predicted):
    return confusion_matrix(labels.cpu(), predicted.cpu())


# Function to compute AUC
def compute_auc(outputs, labels, num_classes):
    # Convert to probabilities with softmax
    probabilities = nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
    labels = labels.cpu().numpy()

    # Calculate AUC for each class and average them
    auc = 0.0
    for i in range(num_classes):
        try:
            auc += roc_auc_score((labels == i).astype(int), probabilities[:, i])
        except ValueError:
            pass  # Handle cases where only one class is present in `y_true` or `y_score`.

    auc /= num_classes
    return auc


# Function to update and save training history
def update_history(history_file, epoch, train_metrics, val_metrics):
    # Prepare the history entry
    epoch_data = {
        'epoch': epoch + 1,
        'train': {k: round(v, 3) for k, v in train_metrics.items()},
        'val': {k: round(v, 3) for k, v in val_metrics.items()}
    }

    # Load existing history if it exists
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history = json.load(file)
    else:
        history = []

    # Add the new epoch data to the history
    history.append(epoch_data)

    # Save the updated history back to the file
    with open(history_file, 'w') as file:
        json.dump(history, file, indent=4)


# Function to update metrics
def update_metrics(outputs, labels, num_classes, top_k, metrics):
    # Compute metrics
    batch_metrics, _ = compute_metrics(outputs, labels, num_classes, top_k)

    # Update metric sums with rounded values
    for key in batch_metrics:
        metrics[key] += round(batch_metrics[key], 3) * labels.size(0)
    metrics['total_samples'] += labels.size(0)


# Function to calculate average metrics
def calculate_average_metrics(metrics, loss):
    avg_metrics = {}
    for key in metrics:
        if key != 'total_samples':
            avg_metrics[key] = round(metrics[key] / metrics['total_samples'], 3)
    avg_metrics['loss'] = round(loss / metrics['total_samples'], 3)
    return avg_metrics


# Function to print metrics for a phase (train/validation)
def print_metrics(phase, metrics, top_k):
    """Prints the average metrics for a specific phase of model training or evaluation."""
    prefix = phase.capitalize()  # Get the prefix 'Train', 'Val', 'Test' from phase
    print(f'{prefix} - Loss: {metrics["loss"]:.3f}, '
          f'Accuracy: {metrics["accuracy"]:.3f}, '
          f'Class Accuracy: {metrics["class_accuracy"]:.3f}, '
          f'Top-{top_k} Accuracy: {metrics["topk_acc"]:.3f}, '
          f'F1 Score: {metrics["f1"]:.3f}, '
          f'Precision: {metrics["precision"]:.3f}, '
          f'Recall: {metrics["recall"]:.3f}, '
          f'AUC: {metrics["auc"]:.3f}')


# Function to evaluate on a given data loader
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

            # Update metrics
            update_metrics(outputs, labels, num_classes, top_k, metrics)

    # Calculate average metrics
    metrics = calculate_average_metrics(metrics, total_loss)

    return metrics


# Training function
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


# Function to plot training and validation metrics from history file
def plot_metrics(history_file):
    with open(history_file, 'r') as file:
        history = json.load(file)

    epochs = [entry['epoch'] for entry in history]
    train_losses = [entry['train']['loss'] for entry in history]
    val_losses = [entry['val']['loss'] for entry in history]
    train_accuracies = [entry['train']['accuracy'] for entry in history]
    val_accuracies = [entry['val']['accuracy'] for entry in history]

    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Evaluation function
def evaluate_model(model, test_loader, num_classes, top_k):
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), num_classes, top_k)

    # Print test performance
    print("Test Performance:")
    print_metrics("test", test_metrics, top_k)

    return test_metrics


# Main execution block
if __name__ == '__main__':
    # Get data transformations
    transform_train, transform_test = get_transforms()

    # Load data
    train_loader, val_loader, test_loader = load_data(
        config['data_dir'],
        config['batch_size'],
        config['val_split'],
        transform_train,
        transform_test
    )

    # Initialize model, loss function, and optimizer
    model = CNN(num_classes=config['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    train_metrics, val_metrics = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config['num_epochs'],
        config['num_classes'],
        config['top_k'],
        config['history_file']
    )

    # Evaluate the model
    test_metrics = evaluate_model(model, test_loader, config['num_classes'], config['top_k'])

    # Print final metrics
    print("Final Training Metrics:", {k: round(v, 3) for k, v in train_metrics.items()})
    print("Final Validation Metrics:", {k: round(v, 3) for k, v in val_metrics.items()})
    print("Final Test Metrics:", {k: round(v, 3) for k, v in test_metrics.items()})


    # Plot metrics from history file
    plot_metrics(config['history_file'])
