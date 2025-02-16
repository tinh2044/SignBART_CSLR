import torch
import os
from safetensors.torch import load_file


safetensors_dir = "./pretrained_models"

for file in os.listdir(safetensors_dir):
    if file.endswith(".safetensors"):
        safetensors_path = f"{safetensors_dir}/{file}"
        state_dict = load_file(safetensors_path)
        torch.save(state_dict, safetensors_path.replace(".safetensors", ".pth"))
        os.remove(safetensors_path)
