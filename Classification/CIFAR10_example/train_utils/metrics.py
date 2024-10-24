import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

# def initialize_metrics():
#     return {
#         'accuracy': 0.0,
#         'class_accuracy': 0.0,
#         'topk_acc': 0.0,
#         'f1': 0.0,
#         'precision': 0.0,
#         'recall': 0.0,
#         'auc': 0.0,
#         'total_samples': 0,
#         'loss' : 10000.0
#     }

def compute_metrics(phase_data, loss, config):

    preds = phase_data['preds']
    scores = phase_data['scores']
    labels = phase_data['labels']

    acc = accuracy_score(labels.cpu(), preds)
    confusion = confusion_matrix(labels.cpu(), preds.cpu())
    class_acc = confusion.diagonal() / confusion.sum(axis=1)
    class_acc= np.mean(class_acc)
    topk_acc = compute_top_k_accuracy(scores, labels, top_k=config['top_k'])
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    auc = compute_auc(scores, labels, config['num_classes'])

    metrics = {
        'acc': round(acc,4),
        'class_acc': round(class_acc,4),
        'topk_acc': round(topk_acc,4),
        'f1': round(f1,4),
        'precision': round(precision,4),
        'recall': round(recall,4),
        'auc': round(auc,4),
        'loss': round((loss / labels.shape[0]),4),
        'samples': labels.shape[0],
    }

    return metrics


def compute_top_k_accuracy(scores, labels, top_k):

    _, topk_preds = torch.topk(scores, top_k, dim=1) # scores.clone().detach() isn't needed - scores.requires_grad = False
    topk_correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_accuracy = topk_correct.sum().float() / labels.size(0)
    return topk_accuracy.item()

def compute_auc(scores, labels, num_classes):
    # probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
    scores = scores.numpy()
    labels = labels.numpy()

    auc = 0.0
    for i in range(num_classes):
        try:
            auc += roc_auc_score((labels == i).astype(int), scores[:, i])
        except ValueError:
            pass

    auc /= num_classes
    return auc


# def calculate_average_metrics(metrics, loss):
#     avg_metrics = {key: round(metrics[key] / metrics['total_samples'], 3) for key in metrics if key != 'total_samples'}
#     avg_metrics['loss'] = round(loss / metrics['total_samples'], 3)
#     return avg_metrics


def print_metrics(phase, metrics, top_k):
    prefix = phase.capitalize()
    print(f'{prefix} - Loss: {metrics["loss"]:.4f}, '
          f'Acc: {metrics["acc"]:.4f}, '
          f'Class Acc: {metrics["class_acc"]:.4f}, '
          f'Top-{top_k} Acc: {metrics["topk_acc"]:.4f}, '
          f'F1 Score: {metrics["f1"]:.4f}, '
          f'Precision: {metrics["precision"]:.4f}, '
          f'Recall: {metrics["recall"]:.4f}, '
          f'AUC: {metrics["auc"]:.4f}')