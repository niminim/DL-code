import os
import json
import re
import torch

def save_best_model(model, epoch, val_acc, best_val_acc, config):
    """
    Save the best model based on validation accuracy. Deletes the previous best model.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    epoch (int): The current epoch.
    val_accuracy (float): The validation accuracy for the current epoch.
    best_val_accuracy (float): The best validation accuracy observed so far.
    config (dict): Dictionary with train config

    Returns:
    str: The path of the saved model.
    float: Updated best validation accuracy.
    """

    models_dir = config['models_dir']
    model_name = config['model_name']

    # Create save directory if it doesn't exist
    os.makedirs(models_dir,exist_ok=True)

    # Save the model if the current validation accuracy is the best seen so far
    if val_acc > best_val_acc:
        # Delete previous best model
        for filename in os.listdir(models_dir):
            if filename.endswith(".pth"):
                os.remove(os.path.join(models_dir, filename))

        # Define the model filename with epoch and validation accuracy
        model_filename = f"{model_name}_epoch_{epoch}_val_acc_{val_acc:.3f}.pth"
        model_path = os.path.join(models_dir, model_filename)

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model: {model_filename}")

        # Update the best validation accuracy
        best_val_acc = val_acc

        return model_path, best_val_acc
    else:
        return None, best_val_acc


def get_best_model_path(directory, metric="val_acc"):
    """
    This function lists all model files in a given directory, extracts the specified metric (e.g., val_acc, val_f1, val_auc)
    from the filenames, sorts the models by the metric value in descending order, and returns the filename of the model
    with the highest metric value.

    Parameters:
        - directory: The path to the directory containing the model files.
        - metric: The metric to sort by (default is "val_acc"). This should match the metric name used in the filenames.

    Returns:
        - The filename of the model with the highest value for the specified metric.
    """

    # Build a dynamic regex pattern based on the provided metric name
    pattern = re.compile(rf"_epoch_(\d+)_({metric})_([0-9.]+)\.pth")

    # List to store tuples of (filename, metric_value)
    models = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Match the pattern to extract the epoch number and metric value from the filename
        match = pattern.search(filename)
        if match:
            # Extract the epoch number from the first capture group
            epoch = int(match.group(1))
            # Extract the metric value (e.g., validation accuracy) from the second capture group
            val_acc = float(match.group(3))
            # Append a tuple of (filename, metric value) to the list
            models.append((filename, val_acc))

    # If there are no models, return None
    if not models:
        return None

    # Sort the models by the metric value in descending order
    models.sort(key=lambda x: x[1], reverse=True)

    best_val_model_path = os.path.join(directory, models[0][0])

    return best_val_model_path

def load_best_val_model(model, device, models_dir, select_metric='val_acc'):
    # loads best val model the the selected metrics

    assert select_metric in os.listdir(models_dir)[0]
    best_val_model_path = get_best_model_path(models_dir, metric=select_metric)
    print(f"\nThe best model's filename: {best_val_model_path.split('/')[-1]}")
    model.load_state_dict(torch.load(best_val_model_path))
    print("Loaded the best val model")
    model.to(device)     # Move the model to the appropriate device (CPU or GPU)
    return model

def delete_all_files(directory):
    # Delete all files in a directory

    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate through all files and directories in the specified directory
        for filename in os.listdir(directory):
            # Construct the full path to the file or directory
            file_path = os.path.join(directory, filename)
            try:
                # Attempt to remove the file
                os.remove(file_path)
            except Exception as e:
                # If an error occurs (e.g., file is a directory or access is denied),
                # print an error message
                print(f"Couldn't delete file: {file_path}. Reason: {e}")


def update_history(history_file, epoch, train_metrics, val_metrics):
    # Update train and val metrics for each epoch in the json file

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