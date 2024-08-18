import os
import json
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
    # Create save directory if it doesn't exist
    models_dir = config['models_dir']
    os.makedirs(models_dir,exist_ok=True)
    model_name = config['model_name']

    # Save the model if the current validation accuracy is the best seen so far
    if val_acc > best_val_acc:
        # Delete previous best model
        for filename in os.listdir(models_dir):
            if filename.endswith(".pth"):
                os.remove(os.path.join(models_dir, filename))

        # Define the model filename with epoch and validation accuracy
        model_filename = f"{model_name}_epoch_{epoch}_valacc_{val_acc:.3f}.pth"
        model_path = os.path.join(models_dir, model_filename)

        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model: {model_filename}")

        # Update the best validation accuracy
        best_val_acc = val_acc

        return model_path, best_val_acc
    else:
        return None, best_val_acc

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