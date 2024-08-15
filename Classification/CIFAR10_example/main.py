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
from Classification.CIFAR10_example.models.build_model import *
from Classification.CIFAR10_example.train_utils.training import train_model, evaluate, print_metrics
from Classification.CIFAR10_example.configs.config import config
from Classification.CIFAR10_example.utils.plots import plot_train_val_loss_acc, plot_multiclass_roc, calc_and_plot_cm_cr
from Classification.CIFAR10_example.utils.neptune_utils import get_neptune_run, add_configs_to_neptune

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch.nn as nn
# torch.set_float32_matmul_precision('high') # could harm results


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if __name__ == '__main__':

    run = get_neptune_run()
    add_configs_to_neptune(run, config)

    # Load data
    train_loader, val_loader, test_loader, class2index = load_data(
        config['data_dir'],
        config['batch_size'],
        config['val_split']
    )

    # Initialize model, loss function, and optimizer
    model = get_model(config, device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.6)

    if os.path.exists(config['history_file']):
        os.remove(config['history_file'])

    # Train the model
    train_metrics, val_metrics = train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                config,
                run)

    # Evaluate the model (need to use best val model and not the last)
    test_scores, test_labels, test_preds, test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), config)
    print_metrics(phase='Test', metrics=test_metrics, top_k=config['top_k'])

    # Plot metrics from history file
    plot_train_val_loss_acc(config['history_file'])
    # Calc Class-A/ucm and plot Class-ROC Curve
    plot_multiclass_roc(test_scores, test_labels, class2index, config, run)
    # Calc and print confusion-matrix and classification-report
    calc_and_plot_cm_cr(test_labels, test_preds, class2index, config, run)

    run.stop()

