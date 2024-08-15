import neptune

def get_neptune_run():
    run = neptune.init_run(
        project="nimrod/dl-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Nzc0MDBlZC1kYTNjLTQwZjItYTdjNi00MzdmOTkyOGRlMDYifQ==",  # your credentials
    name="cifar10_neptune_try"
    )

    return run

def add_configs_to_neptune(run, config):
    run["parameters"] = config

def log_neptune_data(run, train_metrics, val_metrics, optimizer):
    run['metrics/train_loss'].log(train_metrics['loss'])
    run['metrics/train_acc'].log(train_metrics['acc'])
    run['metrics/val_loss'].log(val_metrics['loss'])
    run['metrics/val_acc'].log(val_metrics['acc'])
    run['metrics/lr'].log(optimizer.param_groups[0]['lr'])

    # for k in val_results.keys():
    #     run['metrics/train_miou_' + k.replace(" ","_")].log(train_results[k].miou)
    #     run['metrics/val_miou_' + k.replace(" ","_")].log(val_results[k].miou)

# def update_neptune_run(run, train_metrics, val_metrics):
#     run["train/loss"].append(train_metrics['loss'])
#     run["train/acc"].append(train_metrics['acc'])
#     run["vall/loss"].append(val_metrics['loss'])
#     run["vall/acc"].append(val_metrics['acc'])

