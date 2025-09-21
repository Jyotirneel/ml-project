import os
import json
import random
import shutil
from pathlib import Path

def set_seed(seed=42):
    import torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
