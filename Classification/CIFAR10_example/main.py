import os
import sys

# Determine the project root directory and add it to sys.path
# Navigate up two levels from the current directory
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# print(project_root)
# sys.path.append(project_root)

project_root = os.path.abspath("/home/nim/venv/DL-code")
sys.path.append(project_root)


from Classification.CIFAR10_example.data.datasets import load_data
from Classification.CIFAR10_example.models.cnn import CNN
from Classification.CIFAR10_example.train_utils.training import train_model, evaluate_model
from Classification.CIFAR10_example.configs.config import config
from Classification.CIFAR10_example.utils.plots import plot_metrics
import torch
import torch.optim as optim
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load data
    train_loader, val_loader, test_loader = load_data(
        config['data_dir'],
        config['batch_size'],
        config['val_split']
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