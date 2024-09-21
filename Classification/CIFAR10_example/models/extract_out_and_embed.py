import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

# # Assuming model and input_tensor are already defined and on CUDA

### 1. Wrap the Model

class ModelWithEmbeddings(nn.Module):
    def __init__(self, model, feature_extractor):
        super(ModelWithEmbeddings, self).__init__()
        self.model = model
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # Get the embeddings from the feature extractor
        embeddings = self.feature_extractor(x)['embeddings']

        # Get the original output (classification output)
        out = self.model(x)

        # Return both embeddings and output
        return embeddings, out


# Assuming 'model' is your EfficientNet-B0 model
return_nodes = {
    'global_pool.flatten': 'embeddings'  # Extract embeddings from the global pool layer
}

# Create the feature extractor
feature_extractor = create_feature_extractor(model, return_nodes)

# Create the wrapper model
model_with_embeddings = ModelWithEmbeddings(model, feature_extractor)

# Ensure both model and input_tensor are on CUDA
model_with_embeddings = model_with_embeddings.cuda()
input_tensor = input_tensor.cuda()

# Get the embeddings and original output in one pass
embeddings, out = model_with_embeddings(input_tensor)

##############


### 2. Using Hooks for Single Forward Pass

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


# what's the handle and why we should remove it? (Q)

# The handle in this context refers to the object returned by the register_forward_hook()
# function. When you register a hook on a layer in PyTorch, it returns a handle that allows
# you to manage the hook. Specifically, you can use this handle to remove the hook when
# it's no longer needed, or to keep track of multiple hooks.

# Why should you remove the handle?
# Prevent Accumulation of Hooks: If you don't remove the hook after you've used it,
# it will remain active for every future forward pass. This means every forward pass
# will trigger the hook, which may not be necessary after you've extracted the
# desired embeddings. It can lead to unexpected behavior and memory overhead.

# Avoid Memory Leaks: Hooks add memory overhead because they store references to
# intermediate activations and the hook function. Keeping unnecessary hooks can cause
# memory leaks, especially when running on multiple inputs or during training.

# Performance Considerations: If you leave hooks active, they may unnecessarily
# slow down the forward pass by triggering the hook function every time.
# Removing the hook ensures that only the forward pass runs after you've extracted
# what you need.