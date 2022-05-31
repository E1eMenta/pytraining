import os
import time
from collections import defaultdict
from typing import Optional, Dict, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
            self,
            tb_folder: Optional[str],
            print_freq: int = 100,
            image_freq: int = 1000,
            max_num_images: int = 10,
            exponential_gamma: float = 0.97
    ):
        self.tb_folder = tb_folder
        self.print_freq = print_freq
        self.image_freq = image_freq
        self.max_num_images = max_num_images
        self.gamma = exponential_gamma

        self.loss_buff = defaultdict(float)

        self.val_loss_buff = defaultdict(list)
        self.val_metric_buff = defaultdict(list)

        os.makedirs(tb_folder, exist_ok=True)
        self.writer = SummaryWriter(tb_folder)

        self.start = time.time()
        self.val_image_counter = 0
        self.val_iteration = 0
        self.val_epoch = 0
        self.val_start = 0

    def log_train(
            self,
            epoch: int,
            iteration: int,
            losses: Optional[Dict[str, float]] = None,
            images: Optional[Dict[str, torch.Tensor]] = None
    ):
        if losses is not None:
            for key, value in losses.items():
                if key not in self.loss_buff:
                    self.loss_buff[key] = value

                self.loss_buff[key] = self.gamma * self.loss_buff[key] + (1.0 - self.gamma) * value
                self.writer.add_scalar(f"train_loss/{key}", value, global_step=iteration)

        if iteration % self.print_freq == 0 and iteration > 0:
            rounded_losses = {key: round(value, 4) for key, value in losses.items()}
            print(f"Iter: {iteration} (Epoch: {epoch}). "
                  f"Losses: {rounded_losses}. "
                  f"Time: {round(time.time() - self.start)}.")

        if iteration % self.image_freq == 0 and iteration > 0 and images:
            for key, value in images.items():
                self.writer.add_image(f"train_images/{key}", value, iteration)

    def log_val(
            self,
            epoch: int,
            iteration: int,
            losses: Optional[Dict[str, float]] = None,
            metrics: Optional[Dict[str, float]] = None,
            images: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.val_iteration = iteration
        self.val_epoch = epoch
        self.val_start = time.time()

        if losses is not None:
            for key, value in losses.items():
                self.val_loss_buff[key].append(value)

        if metrics is not None:
            for key, value in metrics.items():
                self.val_metric_buff[key].append(value)

        if images is not None and self.val_image_counter < self.max_num_images:
            for key, value in images.items():
                self.writer.add_image(f"val_images/{key}", value, global_step=self.val_iteration)
            self.val_image_counter += 1

    def end_val(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.val_image_counter = 0

        avg_losses = {key: (sum(value) / len(value)) for key, value in self.val_loss_buff.items() if len(value) > 0}
        avg_metrics = {key: (sum(value) / len(value)) for key, value in self.val_metric_buff.items() if len(value) > 0}

        for key, value in avg_losses.items():
            self.writer.add_scalar(f"val_loss/{key}", value, global_step=self.val_iteration)

        for key, value in avg_metrics.items():
            self.writer.add_scalar(f"val_metric/{key}", value, global_step=self.val_iteration)

        rounded_losses = {key: round(value, 4) for key, value in avg_losses.items()}
        rounded_metrics = {key: round(value, 4) for key, value in avg_metrics.items()}
        print(f"Epoch: {self.val_epoch}. "
              f"Losses: {rounded_losses}. "
              f"Metrics: {rounded_metrics}. "
              f"Time: {round(time.time() - self.val_start)}.")

        return avg_losses, avg_metrics
