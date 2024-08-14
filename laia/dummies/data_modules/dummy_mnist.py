import pytorch_lightning as pl
import torch
import torchvision

from laia import __root__
from laia.data.transforms.vision import ToImageTensor


class DummyMNIST(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.root = __root__ / "datasets"
        self._train_transforms = ToImageTensor()
        self._val_transforms = ToImageTensor()
        self.tr_ds = None
        self.va_ds = None

    @property
    def train_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to train dataset."""
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t):
        self._train_transforms = t

    @property
    def val_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to validation dataset."""
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t):
        self._val_transforms = t

    @property
    def test_transforms(self):
        """Optional transforms (or collection of transforms) you can apply to test dataset."""
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t):
        self._test_transforms = t

    def prepare_data(self):
        torchvision.datasets.MNIST(self.root, train=True, download=True)
        torchvision.datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage):
        self.tr_ds = torchvision.datasets.MNIST(
            self.root,
            train=stage == "fit",
            transform=self.train_transforms,
        )
        self.va_ds = torchvision.datasets.MNIST(
            self.root,
            train=stage != "fit",
            transform=self.val_transforms,
        )

    def collate_fn(self, batch):
        x = torch.stack([a for a, b in batch])
        y = [[b] for a, b in batch]
        return x, y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.tr_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.va_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
