import runpod

version = runpod.version.get_version()

print(f"RunPod version number: {version}")

import os
RUNPOD_API_KEY = "rpa_3B8LZ7IDLWFFM0JXIGZT5X0TL5F05PW8M8RFHG911ymy7s"
RUNPOD_API_KEY = "rpa_3FJCFGT1NA0A45V9J76RKNRJ74GXRWGDW6OIWB3Y1bf0zb"
runpod.api_key = os.getenv(RUNPOD_API_KEY)

import torch

torch.cuda.is_available()

# Fetching all available endpoints
endpoints = runpod.get_endpoints()

# Displaying the list of endpoints
print(endpoints)