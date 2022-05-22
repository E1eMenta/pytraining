from typing import List

import torch
import torchvision.transforms.functional as F


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    assert len(mean) == len(std)

    demean = [-mean_el / std_el for mean_el, std_el in zip(mean, std)]
    destd = [1.0 / std_el for std_el in std]
    return F.normalize(tensor, demean, destd, inplace)
