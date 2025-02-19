config = {
    'model_name': 'mobilenetv3', # cnn, mobilenetv3, mixnet_s, efficientnet_b0, efficientnet_b1, efficientnet_b2
    'batch_size': 256,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'use_amp': True,
    'data_dir': '/home/nim/cifar10_project/data',
    'num_classes': 10,
    'val_split': 0.2,
    'top_k': 5,
    'history_file': '/home/nim/cifar10_project/training_history.json',
    'models_dir': '/home/nim/cifar10_project/models',
    'plots_dir': '/home/nim/cifar10_project/plots',
    'save_to_neptune': False
}

config['models_dir'] = f"/home/nim/cifar10_project/models/{config['model_name']}"

