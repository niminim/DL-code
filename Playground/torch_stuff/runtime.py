import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, resnet101, resnet152

# Hyperparameters
params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 5,
    "model_size": 3, # 1 - resnet18, 2 - resnet50, 3 - resnet101, 4 - resnet 152
    "use_amp": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
def get_model(model_size, device):
    if model_size==1:
        model = resnet18(weights=None, num_classes=10)
    elif model_size == 2:
        model = resnet50(weights=None, num_classes=10)
    elif model_size == 3:
        model = resnet101(weights=None, num_classes=10)  # Using ResNet152 without pretrained weights
    elif model_size == 4:
        model = resnet152(weights=None, num_classes=10)  # Using ResNet152 without pretrained weights
    model = model.to(device)
    return model

# Generate random images of size [batch_size, 3, 320, 320] and random labels
num_samples = 1000
image_size = (3, 320, 320)
random_images = torch.randn(num_samples, *image_size)
random_labels = torch.randint(0, 10, (num_samples,))  # Assuming 10 classes

# Create TensorDataset and DataLoader
dataset = TensorDataset(random_images, random_labels)
train_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
val_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

model = get_model(params["model_size"], device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])



# Function for training one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_train_loss = 0.0
    forward_timings = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for inputs, targets in train_loader:
        starter.record()
        inputs, targets = inputs.to(device), targets.to(device)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        cur_time = starter.elapsed_time(ender)
        forward_timings.append(cur_time)

        if use_amp:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    return epoch_train_loss, sum(forward_timings)/1000.0


# Function for validation
def validate(model, val_loader, criterion, device):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    return epoch_val_loss


# Training and Validation Loop with Time Measurement
scaler = torch.amp.GradScaler() if params['use_amp'] else None

for epoch in range(params["epochs"]):
    start_time = time.time()  # Start the timer at the beginning of the epoch
    timings = []
    # Train for one epoch

    # Record the start event for the epoch
    starter_epoch, ender_epoch = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter_epoch.record()

    epoch_train_loss, sum_forward_timings = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, params['use_amp'])

    # Record the end event for the epoch
    ender_epoch.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    epoch_cur_time = starter_epoch.elapsed_time(ender_epoch)/1000.0


    timing = time.time() - start_time  # Calculate elapsed time
    timings.append(timing)
    # Validate after the epoch
    epoch_val_loss = validate(model, val_loader, criterion, device)

    # End the timer at the end of the epoch
    epoch_time = time.time() - start_time  # Calculate elapsed time

    print(
        f"Epoch [{epoch + 1}/{params['epochs']}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Time: {epoch_time:.2f} seconds")
    print(f"epoch_cur_time GPU time: {epoch_cur_time:.3f} seconds")
    print(f"forward GPU time: {sum_forward_timings:.3f} seconds")
    print('***')
