import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np


def compute_metrics(outputs, labels, num_classes, top_k=5):
    _, predicted = torch.max(outputs.data, 1)

    accuracy = accuracy_score(labels.cpu(), predicted.cpu())
    confusion = confusion_matrix(labels.cpu(), predicted.cpu())
    class_accuracy = confusion.diagonal() / confusion.sum(axis=1)
    class_accuracy = np.mean(class_accuracy)

    topk_acc = compute_top_k_accuracy(outputs, labels, k=top_k)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    precision = precision_score(labels.cpu(), predicted.cpu(), average='weighted', zero_division=0)
    recall = recall_score(labels.cpu(), predicted.cpu(), average='weighted', zero_division=0)
    auc = compute_auc(outputs, labels, num_classes)

    metrics = {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'topk_acc': topk_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    return metrics


def compute_top_k_accuracy(outputs, labels, k=5):
    _, topk_preds = torch.topk(outputs, k, dim=1)
    topk_correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_accuracy = topk_correct.sum().float() / labels.size(0)
    return topk_accuracy.item()


def compute_auc(outputs, labels, num_classes):
    probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
    labels = labels.cpu().numpy()

    auc = 0.0
    for i in range(num_classes):
        try:
            auc += roc_auc_score((labels == i).astype(int), probabilities[:, i])
        except ValueError:
            pass

    auc /= num_classes
    return auc