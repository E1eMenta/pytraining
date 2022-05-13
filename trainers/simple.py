from typing import Callable, Dict, Optional, List, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleTrainer:
    def __init__(
            self,
            model: nn.Module,
            criterion: Callable[..., torch.Tensor],
            metric: Optional[Callable[..., Dict[str, float]]],
            optimizer: optim.Optimizer,

            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            steps_per_epoch: int,
            max_epoch: int,

            input_key: str = "input",
            target_key: str = "target",
            storage: Optional[Any] = None,
            logger: Optional[Any] = None,
            scheduler: Optional[Any] = None,

            device: str = "cuda",
            visualise_keys: Optional[List[str]] = None
    ):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.steps_per_epoch = steps_per_epoch
        self.max_epoch = max_epoch

        self.input_key = input_key
        self.target_key = target_key
        self.scheduler = scheduler
        self.storage = storage
        self.logger = logger
        self.device = device
        self.visualise_keys = [] if visualise_keys is None else visualise_keys

        self.epoch = 0
        self.iteration = 0

    def train(self):
        self.model.train()

        for i in range(self.steps_per_epoch):
            data = self.get_data()
            inputs = data[self.input_key].to(self.device)
            target = data[self.target_key].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            data["output"] = pred

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step(loss.item())

            if self.logger is not None:
                self.logger.log_train(
                    iteration=self.iteration,
                    losses={'main': loss.item()},
                    images={key: data[key] for key in self.visualise_keys},
                )

            self.iteration += 1

    def validate(self):
        self.model.eval()

        for data in self.val_dataloader:
            inputs = data[self.input_key].to(self.device)
            target = data[self.target_key].to(self.device)

            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            data["output"] = pred

            metric = self.metric(pred, target) if self.metric is not None else None
            self.logger.log_val(
                iteration=self.iteration,
                losses={'main': loss.item()},
                metrics=metric,
                images={key: data[key] for key in self.visualise_keys}
            )

        _, avg_metrics = self.logger.end_val()
        self.storage.save(
            self.epoch,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
            avg_metrics
        )

        self.epoch += 1

    def run(self):
        for epoch in range(self.max_epoch):
            self.train()
            with torch.no_grad():
                self.validate()
