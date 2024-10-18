import gc

import torch

BatchType = dict[str, torch.Tensor]


def clear_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
