import os.path

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import json
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system


def plot_train_val_loss_acc(history_file):
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


def plot_multiclass_roc(scores, labels, class2index,config, run):
    """
    Plots ROC curves and computes AUC for a multiclass classification problem.

    Parameters:
    val_scores (torch.Tensor): The predicted scores (probabilities or logits) with shape [num_samples, num_classes].
    val_labels (torch.Tensor): The true labels with shape [num_samples, 1].
    num_classes (int): The number of classes.

    Returns:
    None
    """

    num_classes = config['num_classes']
    os.makedirs(config['plots_dir'],exist_ok=True)
    roc_file_path = os.path.join(config['plots_dir'],'roc_curve.png')

    # Create a new dictionary by swapping keys and values
    index2class = {value: key for key, value in class2index.items()}

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

    print('')
    # Example: Printing the AUC values
    for i in range(num_classes):
        print(f'Class {index2class[i]} AUC: {roc_auc[i]:.2f}')


    # Plot all ROC curves
    print('')
    print('Plot all Class-ROC curves:')
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {index2class[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(roc_file_path)
    run["test_reports/roc_curve"].upload(roc_file_path)
    plt.show()



def calc_and_plot_cm_cr(labels, preds, class2index, config, run):

    os.makedirs(config['plots_dir'],exist_ok=True)
    cm_file_path = os.path.join(config['plots_dir'],'confusion_matrix.png')
    cr_file_path = os.path.join(config['plots_dir'],'classification_report.png')


    # Create a new dictionary by swapping keys and values
    index2class = {value: key for key, value in class2index.items()}

    # Confusion matrix
    cm = confusion_matrix(labels.cpu(), preds.cpu())

    # Convert confusion matrix to DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=[index2class[i] for i in range(len(cm))], # or index=list(class2index.keys())
                         columns=[index2class[i] for i in range(len(cm))]) # or index=list(class2index.keys())

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(cm_file_path)
    run["test_reports/confusion_matrix"].upload(cm_file_path)

    plt.show()


    # classification report
    cr = classification_report(preds.cpu(), labels.cpu(),target_names=[index2class[i] for i in range(len(class2index))])
    cr_dict = classification_report(preds.cpu(), labels.cpu(),target_names=[index2class[i] for i in range(len(class2index))],output_dict=True)
    cr_df = pd.DataFrame(cr_dict)

    # Plot the classification report
    plt.figure(figsize=(10, 6))
    sns.heatmap(cr_df.iloc[:-1, :].T, annot=True, cmap='Blues', cbar=False)
    plt.title('Classification Report')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.savefig(cr_file_path)
    run["test_reports/classification_report"].upload(cr_file_path)
    plt.show()

    print('')
    print('cm')
    print(cm)
    print('cr')
    print(cr)
    return cm_df, cr





