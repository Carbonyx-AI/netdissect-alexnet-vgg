"""Network Dissection (Bau et al., CVPR 2017) re-implemented in PyTorch.

Paths can be changed with environment variables, see README.
"""
import os

import torch

WORK = os.environ.get("ND_WORK", "work")
DATA = os.environ.get("ND_DATA", os.path.join(WORK, "broden"))
WEIGHTS = os.environ.get("ND_WEIGHTS", WORK)
IMAGENETTE = os.environ.get("ND_IMAGENETTE", os.path.join(WORK, "imagenette2-320"))
TRAINED = os.environ.get("ND_TRAINED", os.path.join(WORK, "trained"))
OUT = os.environ.get("ND_OUT", "results")


def device(gpu_mem_gb=10.0):
    """'cuda' if a GPU is there (and limit our share of its memory), otherwise 'cpu'."""
    if os.environ.get("ND_DEVICE") == "cpu" or not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_gb / total))
    torch.backends.cudnn.benchmark = True
    return "cuda"
