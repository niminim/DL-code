import yaml


def read_cnfg(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


#####

if __name__ == "__main__":

    from torch import nn as nn
    from torchvision import models
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR


    model = models.efficientnet_b1(weights='IMAGENET1K_V1', progress=True)
    model.features[0][0] = nn.Conv2d(in_channels=config['model']['in_channels'],
                                     out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(1280, config['model']['num_classes'])

    # Assuming model is already defined
    optimizer = optim.SGD(model.parameters(),
                          lr=config['train']['lr'],
                          momentum=config['train']['optim_params']['momentum'],
                          nesterov=config['train']['optim_params']['nesterov'])


    scheduler = StepLR(optimizer,
                       step_size=config['train']['lr_decay']['step_size'],
                       gamma=config['train']['lr_decay']['rate'])


    ### Loss function
    import torch.nn as nn

    # Create the loss function from the config
    loss_function_name = config['train']['loss_fn']
    criterion = getattr(nn, loss_function_name)()