import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision.models import resnet18

from logger.simple import Logger
from metrics.classification import Accuracy
from trainers.simple import SimpleTrainer


class DictWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, target = self.dataset[item]
        return {"image": img, "label": target}


class Config:
    def __init__(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = DictWrapper(torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train))
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = DictWrapper(torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test))
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        model = resnet18(pretrained=True)
        model.to("cuda")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        logger = Logger('/cache/tmp/cifar/logs')
        metric = Accuracy()

        self.trainer = SimpleTrainer(
            model=model,
            criterion=criterion,
            metric=None,
            optimizer=optimizer,
            train_dataloader=trainloader,
            val_dataloader=testloader,
            steps_per_epoch=200,
            max_epoch=200,
            input_key="image",
            target_key="label",
            storage=None,
            logger=logger,
            scheduler=scheduler,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            device="cuda",
            visualise_keys=["image"]
        )

    def run(self):
        self.trainer.run()
