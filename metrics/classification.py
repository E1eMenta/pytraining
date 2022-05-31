from typing import Dict

import torch


class Accuracy:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        _, predicted = pred.max(1)

        total = target.size(0)
        correct = predicted.eq(target).sum().item()

        acc = float(100. * correct / total)
        return {'accuracy': acc}
