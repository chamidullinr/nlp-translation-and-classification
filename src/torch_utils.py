from typing import Iterable

import numpy as np
import torch


def torch_isin(element: torch.Tensor, test_elements: Iterable):
    # note - torch.isin will be available in PyTorch=1.10
    cond = np.isin(element.cpu(), test_elements)
    cond = torch.Tensor(cond).to(torch.bool)
    return cond
