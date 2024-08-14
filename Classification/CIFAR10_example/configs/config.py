config = {
    'model_name': 'cnn', # cnn, mobilenetv3, mixnet_s, efficientnet_b0, efficientnet_b1
    'batch_size': 256,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'use_amp': False,
    'data_dir': '/home/nim/data',
    'num_classes': 10,
    'val_split': 0.2,
    'top_k': 5,
    'history_file': '/home/nim/data/training_history.json',
    'models_dir': '/home/nim/cifar10_models'
}
