from torchvision.models.feature_extraction import create_feature_extractor

import torch

def extract_embed(config, model):
    model_name = config['model_name'].lower()

    if model_name == 'mobilenetv3':
        model = model._orig_mod  # Access the original model
        return_nodes = {
            'avgpool': 'embeddings'  # Extract embeddings from the avgpool layer
        }

    elif model_name == 'efficientnet_b0':
        return_nodes = {
            'global_pool.flatten': 'embeddings'  # Extract embeddings from the avgpool layer
        }

    feature_extractor = create_feature_extractor(model, return_nodes)

    # Example input tensor
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Extract embeddings
    features = feature_extractor(input_tensor)
    embeddings = features['embeddings']

    # Reshape embeddings if needed
    embeddings = embeddings.view(embeddings.size(0), -1)
    print(embeddings.shape)  # Outputs (batch_size, embed_size)

    return embeddings





# Iterate over the named children (higher-level layers)
for name, layer in model.named_children():
    print(f"Layer name: {name}, Layer: {layer}")

# Iterate over all named modules (including sub-layers)
for name, module in model.named_modules():
    print(f"Module name: {name}, Module: {module}")