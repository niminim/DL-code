import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system

import json

def plot_metrics(history_file):
    with open(history_file, 'r') as file:
        history = json.load(file)

    epochs = [entry['epoch'] for entry in history]
    train_losses = [entry['train']['loss'] for entry in history]
    val_losses = [entry['val']['loss'] for entry in history]
    train_accuracies = [entry['train']['acc'] for entry in history]
    val_accuracies = [entry['val']['acc'] for entry in history]

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


def plot_multiclass_roc(scores, labels, num_classes):
    """
    Plots ROC curves and computes AUC for a multiclass classification problem.

    Parameters:
    val_scores (torch.Tensor): The predicted scores (probabilities or logits) with shape [num_samples, num_classes].
    val_labels (torch.Tensor): The true labels with shape [num_samples, 1].
    num_classes (int): The number of classes.

    Returns:
    None
    """
    # Reshape val_labels to be a 1D tensor
    labels = labels.view(-1)

    # Binarize the labels (One-vs-Rest)
    val_labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    # Initialize dictionaries to store fpr, tpr, and auc for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(val_labels_bin[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Example: Printing the AUC values
    for i in range(num_classes):
        print(f'Class {i} AUC: {roc_auc[i]:.2f}')


    # Plot all ROC curves
    print('')
    print('Plot all Class-ROC curves:')
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass')
    plt.legend(loc="lower right")
    plt.show()









