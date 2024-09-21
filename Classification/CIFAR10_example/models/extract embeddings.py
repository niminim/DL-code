import torch
from torchvision.models.feature_extraction import create_feature_extractor

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

    elif model_name == 'efficientnet_b1':
        return_nodes = {
            'avgpool': 'embeddings'  # Extract embeddings from the avgpool layer
        }

    elif model_name == 'efficientnet_b2':
        return_nodes = {
            'avgpool': 'embeddings'  # Extract embeddings from the avgpool layer
        }

    elif model_name == 'mixnet_s':
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



def print_named_children(model):
    # Iterate over the named children (higher-level layers)
    for name, layer in model.named_children():
        print(f"Layer name: {name}, Layer: {layer}")

def print_named_modules(model):
    # Iterate over all named modules (including sub-layers)
    for name, module in model.named_modules():
        print(f"Module name: {name}, Module: {module}")



###################

# Assuming model and input_tensor are already on CUDA

# This will store the embeddings during the forward pass
embeddings = None

# Hook function to capture the embeddings
def hook_fn(module, input, output):
    global embeddings
    embeddings = output

# Register a forward hook on the layer you're interested in
# In this case, 'global_pool.flatten' is the layer you want to extract embeddings from
handle = model.global_pool.flatten.register_forward_hook(hook_fn)

# Now, when you run the model, the embeddings will be captured by the hook
out = model(input_tensor)

# At this point, 'embeddings' contains the output from the global_pool.flatten layer
# and 'out' contains the final output of the model

# Don't forget to remove the hook after you're done
handle.remove()

# Now you have both 'embeddings' and 'out'