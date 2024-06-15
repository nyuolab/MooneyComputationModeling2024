# This file is part of Mooney computational modeling project.
#
# Mooney computational modeling project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mooney computational modeling project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mooney computational modeling project. If not, see <https://www.gnu.org/licenses/>.

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import yaml
import click
from pathlib import Path

import lightning as L

import torch
from torch.optim import AdamW

from src.model import (
    TopDownPerceiver,
    ConditionedPerceiverConfig,
)
from src.datamodules.standard import ImageNetDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger


def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=0.0001)
    return {
        "optimizer": optimizer,
    }

@click.command()
@click.option('--seed', type=int, required=True, help="Random seed.")
@click.option('--backbone_name', type=str, required=True, help="Name of the backbone.")
@click.option('--config_file', type=str, required=True, help="Training config.")
@click.option('--checkpoint', type=str, required=True, help="Output checkpoint path.")
def main(seed, config_file, checkpoint, backbone_name):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    full_backbone_name = "/".join([config["model_prefix"], backbone_name])

    # Seed
    L.seed_everything(seed)
    torch.set_float32_matmul_precision('medium')

    backbone_channels = config["backbone_channels"][backbone_name]

    # Add the optimizer method
    setattr(TopDownPerceiver, "configure_optimizers", configure_optimizers)

    # Get the data
    data = ImageNetDataModule(
        dataset_name=config["dataset"],
        cache_dir=None if config["cache_dir"] == "None" else config["cache_dir"],
        train_split_name=config["train_split_name"],
        val_split_name=config["val_split_name"],
        train_batch_size=config["train_batchsize"],
        eval_batch_size=config["eval_batchsize"],
        source_model_name=full_backbone_name,
        limit_train_data=config["n_subset_train"],
        limit_val_data=config["n_subset_eval"],
        dataloader_num_workers=config["workers"],
    )

    model_config = ConditionedPerceiverConfig(
        backbone_model_name=full_backbone_name,
        num_backbone_channels=backbone_channels,
        batch_mixing=config["batch_mixing"],
        temporal_repeats=config["temporal_repeats"],
        num_condition_tokens=config["n_condition_tokens"],
        cross_attention_heads=config["cross_attention_heads"],
        r=config["lora_r"],
        bias="none",
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.1,
        target_modules=config["lora_targets"],
        modules_to_save=["patch_embedding"]
    )

    # Get the run ID
    logger = WandbLogger(project=config["project_name"], entity="Aceticia", tags=config["tags"], name=f"{backbone_name}-seed-{seed}")

    # Create the model from config
    lit_model = TopDownPerceiver.create(model_config)

    # The checkpoint to store is actually one up
    checkpoint = Path(checkpoint)
    checkpoint_parent = checkpoint.parent

    trainer = L.Trainer(
        gradient_clip_val=0.5,
        accelerator="cpu" if config["use_cpu"] else "auto",
        num_nodes=1 if config["use_cpu"] else config["n_nodes"],
        devices="auto",
        strategy="auto",
        max_epochs=config["epochs"],
        precision="16-mixed" if config["fp16"] else 32,
        accumulate_grad_batches=config["accumulate_steps"],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_post_acc", patience=5, mode="max"),
            ModelCheckpoint(monitor="val_post_acc", mode="max", save_top_k=1, save_last=True, verbose=True, filename="best_model", dirpath=checkpoint_parent)
        ],
        logger=logger
    )
    trainer.fit(lit_model, datamodule=data)

if __name__ == "__main__":
    main()
