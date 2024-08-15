import json
import torch
import torch.nn as nn

from Classification.CIFAR10_example.train_utils.metrics import *
from Classification.CIFAR10_example.train_utils.general_utils import *
from Classification.CIFAR10_example.utils.neptune_utils import update_neptune_run, log_neptune_data


# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, run):

    top_k = config['top_k']
    num_classes = config['num_classes']
    num_epochs = config['num_epochs']
    models_dir = config['models_dir']
    history_file = config['history_file']
    use_amp = config['use_amp']

    # Initialize GradScaler if using AMP
    scaler = torch.amp.GradScaler() if use_amp else None

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
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called, otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                # Forward pass
                outputs = model(images)  # (cuda, requires_grad)
                loss = criterion(outputs, labels)  # (cuda, requires_grad)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            scores = torch.nn.functional.softmax(outputs, dim=1) # (cuda, requires_grad)
            _, preds = torch.max(outputs.data, 1) #  basically it's - descending scores, preds, but we don't want to overide the scores  (preds not on cuda)

            labels = torch.unsqueeze(labels.cpu(), dim=1) # reshape
            train_labels = torch.cat((train_labels, labels), dim=0)
            preds = torch.unsqueeze(preds, dim=1).cpu() # reshape
            train_preds = torch.cat((train_preds, preds), dim=0)
            train_scores = torch.cat((train_scores, scores.detach().cpu()), dim=0)

            total += labels.size(0)
            correct += (preds == labels.cpu()).sum().item()

        scheduler.step()

        # Calculate metrics
        train_metrics = compute_metrics(train_preds, train_scores, train_labels, running_loss, config)
        val_scores, val_labels, val_preds, val_metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), config)


        # Print metrics at the end of each epoch
        print_metrics(phase='Train', metrics=train_metrics, top_k=top_k)
        print_metrics(phase='Val', metrics=val_metrics, top_k=top_k)

        # Update and save training history
        update_history(history_file, epoch, train_metrics, val_metrics)

        # Update neptune logging
        update_neptune_run(run, train_metrics, val_metrics)
        log_neptune_data(run, train_metrics, val_metrics, optimizer)

        # Save the best model based on validation accuracy
        _, best_val_acc = save_best_model(model, epoch, val_metrics['acc'], best_val_acc, models_dir, config['model_name'])
        
    return train_metrics, val_metrics


def evaluate(model, loader, criterion, config):

    use_amp = config['use_amp']

    val_scores = torch.empty(0, config['num_classes'])
    val_preds = torch.empty(0, 1)
    val_labels = torch.empty(0, 1)

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
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

    return val_scores, val_labels, val_preds, metrics


def update_history(history_file, epoch, train_metrics, val_metrics):
    epoch_data = {
        'epoch': epoch + 1,
        'train': {k: round(v, 4) for k, v in train_metrics.items()},
        'val': {k: round(v, 4) for k, v in val_metrics.items()}
    }

    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history = json.load(file)

    history.append(epoch_data)

    with open(history_file, 'w') as file:
        json.dump(history, file, indent=4)



