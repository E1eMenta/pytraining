import shutil
from pathlib import Path
from typing import Union, Dict

import torch
from torch import nn


class Storage:
    def __init__(self, save_folder: Union[str, Path], save_freq: int = 1):
        self.save_folder = Path(save_folder)

        if self.save_folder.exists():
            print(f"Clear storage in folder: {self.save_folder}")
            shutil.rmtree(self.save_folder)

        self.save_folder.mkdir(parents=True)
        self.save_freq = save_freq

    def save(self, epoch: int, iteration: int, modules: Dict[str, nn.Module], metric: Dict[str, float]):
        if epoch % self.save_freq == 0:
            epoch_path = self.save_folder / str(epoch)
            epoch_path.mkdir()
            for module_name, module in modules.items():
                torch.save(module.state_dict(), epoch_path / module_name)
