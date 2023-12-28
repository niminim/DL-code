from datetime import datetime

import torch
import torch.nn.functional as F

from Classification.OCT_Classification.configs.read_configs import read_cnfg
from Classification.OCT_Classification.logs.logging_helper import get_logger
from Classification.OCT_Classification.data_utils.dataloader_from_folders import get_datasets_and_dataloaders

from Classification.OCT_Classification.train_utils import *
from Classification.OCT_Classification.test_utils import *

# print(os.getcwd()) # get current working directory
# sys.path.append('/home/nim/venv/DL-code/Classification/OCT_Classification')
# os.chdir('/home/nim/venv/DL-code/Classification')

config_path = '/home/nim/venv/DL-code/Classification/OCT_Classification/configs/config1.yaml'  # Path to your YAML config file
config = read_cnfg(config_path)
logger = get_logger(config)


###### Train from folder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model, criterion, optimizer, scheduler = get_model_loss_optim(config, device)
train_dataset, val_dataset, train_loader, val_loader = get_datasets_and_dataloaders(config)

start_time = datetime.now()
logger.info(f'Training started at {start_time}')

for epoch in range(1,20+1):  # loop over the dataset multiple times
    train_loss, train_acc,  train_probs, train_predict = train_epoch(epoch, model, optimizer, scheduler, criterion, train_loader, device, logger)

print('Finished Training')
end_time = datetime.now()
logger.info(f'Training finished at {end_time}')
# close_logger(logger)

probs_and_labels, val_preds, true_labels = evaluate(model, val_dataset, val_loader, device)
cm, cr = get_metrics(val_preds, true_labels, val_dataset)
####


############# ###### Train from CSV
from sklearn.preprocessing import LabelEncoder

model, criterion, optimizer, scheduler = get_model_loss_optim(config, device)

for epoch in range(40):  # loop over the dataset multiple times
    print(f'epoch: {epoch}')
    train_loss, train_acc,  train_probs, train_predict = train_epoch_csv(epoch, model, optimizer, scheduler, criterion, train_loader, device, logger)

print('Finished Training')
logger.info('Training completed!')


correct, total = 0, 0
all_preds = torch.empty(0,).to(device)
all_probs = torch.empty(0, 4).to(device) # when creating dataset from CSV there's no .classes

model.eval()
with torch.no_grad():
    for data in val_loader:
        for i, (img, label, img_data) in enumerate(val_loader, 0):
            label_encoder = LabelEncoder()
            label_encoder.fit(label)
            numerical_labels = torch.Tensor(label_encoder.transform(label)).long()
            inputs, labels = img.to(device), numerical_labels.to(device)  # inputs.dtype and labels.dtype - torch.int64

            outputs = model(inputs)
            val_probs = torch.nn.functional.softmax(outputs, dim=1)[:,:4]

            # the class with the highest energy is what we choose as prediction
            val_scores, val_predicted = torch.max(outputs.data, 1)
            all_preds = torch.cat((all_preds, val_predicted),axis=0)
            all_probs = torch.cat((all_probs, val_probs),axis=0)
            total += labels.size(0)
            correct += (val_predicted == labels).sum().item()

print(f'Accuracy of the model on the val-set images: {100 * (correct/total):.2f} %')
print('val_dataset.targets: ',val_dataset.targets)
print('all_preds: ',all_preds)

true_labels = torch.Tensor(val_dataset.targets).reshape(len(val_dataset),1)
final = torch.cat((all_probs, true_labels.to(device)), dim=1)
final = torch.round(final, decimals=3)


cm = confusion_matrix(true_labels.cpu(), all_preds.cpu())
cr = classification_report(true_labels.cpu(), all_preds.cpu(), target_names=list(val_dataset.class_to_idx.keys()))

print('cm')
print(cm)
print('cr')
print(cr)