# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torchsummary import summary

model = models.resnet18().cuda()
summary(model, (3, 224, 224)) # summary moves the input tensor to the GPU


###### Using profiler to analyze execution time

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

### CPU
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


### CPU + GPU
# (Note: the first use of CUDA profiling may bring an extra overhead.)
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

###### Using profiler to analyze memory consumption

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


### Using tracing functionality

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("/home/nim/trace.json")

# You can examine the sequence of profiled operators and CUDA kernels in Chrome trace viewer (chrome://tracing)
# (go to the site and load the json file)

### Examining stack traces
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    model(inputs)

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

### Using profiler to analyze long-running jobs

from torch.profiler import schedule

my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()




# https://pytorch.org/tutorials/beginner/profiler.html  (Profiling your PyTorch Module)
# https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
