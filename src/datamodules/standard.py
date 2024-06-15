from functools import partial

import numpy as np
import lightning as L

import os

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoImageProcessor


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


def transform(preprocessor, input_dict):
    gray_image = input_dict["image"].convert("L")
    binary_image = input_dict["image"].convert("L")

    # Get a random threshold between 50 and 255-50
    threshold = np.random.randint(50, 255 - 50)

    # Threshold the binary image
    binary_image = binary_image.point(lambda p: 255 if p > threshold else 0)

    gray_image = preprocessor(gray_image)["pixel_values"]
    binary_image = preprocessor(binary_image)["pixel_values"]

    gray_image = gray_image[0]
    binary_image = binary_image[0]

    return {
        "label": input_dict["label"],
        "pixel_values": gray_image,
        "binarized_pixel_values": binary_image
    }


class ImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        cache_dir: str,
        train_split_name: str,
        val_split_name: str,
        train_batch_size: int,
        eval_batch_size: int,
        source_model_name: str,
        dataloader_num_workers: int,
        limit_train_data: int = -1,
        limit_val_data: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()

        preprocessor = AutoImageProcessor.from_pretrained(source_model_name, return_tensors="pt")

        self.transform = partial(transform, preprocessor)

    def prepare_data(self):
        load_dataset(self.hparams.dataset_name, split=self.hparams.train_split_name, cache_dir=self.hparams.cache_dir)
        load_dataset(self.hparams.dataset_name, split=self.hparams.val_split_name, cache_dir=self.hparams.cache_dir)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.imagenet_train = DatasetWrapper(
                load_dataset(
                    self.hparams.dataset_name,
                    cache_dir=self.hparams.cache_dir,
                    split=self.hparams.train_split_name,
                ),
                self.transform
            )
            self.imagenet_val = DatasetWrapper(
                load_dataset(
                    self.hparams.dataset_name,
                    cache_dir=self.hparams.cache_dir,
                    split=self.hparams.val_split_name
                ),
                self.transform
            )

            # Limit data
            if self.hparams.limit_train_data > 0:
                self.imagenet_train = Subset(self.imagenet_train, list(range(self.hparams.limit_train_data)))
            if self.hparams.limit_val_data > 0:
                self.imagenet_val = Subset(self.imagenet_val, list(range(self.hparams.limit_val_data)))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.dataloader_num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.dataloader_num_workers,
            persistent_workers=True,
        )
