import torch
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, val_dataset, val_loader, device):
    correct = 0
    total = 0

    val_preds = torch.empty(0, ).to(device)
    all_probs = torch.empty(0, len(val_dataset.classes)).to(device)  # when creating dataset from CSV there's no .classes

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)  # inputs.dtype - torch.float32, labels.dtype - torch.int64

            outputs = model(inputs)
            val_scores, val_predicted = torch.max(outputs.data, 1)
            val_preds = torch.cat((val_preds, val_predicted), axis=0)

            val_probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs = torch.cat((all_probs, val_probs), axis=0)
            correct += (val_predicted == labels).sum().item()
            total += labels.size(0)
    print(f'Accuracy of the model on the val-set images: {100 * (correct / total):.2f} %')

    true_labels = torch.Tensor(val_dataset.targets).reshape(len(val_dataset), 1)
    probs_and_labels = torch.cat((all_probs, true_labels.to(device)), dim=1)
    probs_and_labels = torch.round(probs_and_labels, decimals=3)

    return probs_and_labels, val_preds, true_labels


def get_metrics(val_preds, true_labels, val_dataset):

    cm = confusion_matrix(val_preds.cpu(), true_labels.cpu())
    cr = classification_report(val_preds.cpu(), true_labels.cpu(), target_names=list(val_dataset.class_to_idx.keys()))

    print('cm')
    print(cm)
    print('cr')
    print(cr)
    return cm, cr
