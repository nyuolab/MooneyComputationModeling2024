import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Optional

import lightning as L

from src.model.perceiver_model import (
    ConditionedPerceiverConfig,
    TopDownPerceiverClassifier
)

import torch
import torch.nn as nn
import torchmetrics as tm


class LitPerceiverIO(L.LightningModule):
    def __init__(
        self,
        backbone_model_name: str,
        num_backbone_channels: int,
        batch_mixing: int,
        temporal_repeats: int,
        num_condition_tokens: int,
        cross_attention_heads: int,
        r: int,
        bias: str,
        lora_alpha: float,
        lora_dropout: float,
        target_modules: list[str],
        modules_to_save: list[str],
        params: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    @property
    def backend_model(self):
        return self.model

    @classmethod
    def create(cls, config: ConditionedPerceiverConfig):
        return cls(
            backbone_model_name=config.backbone_model_name,
            num_backbone_channels=config.num_backbone_channels,
            batch_mixing=config.batch_mixing,
            temporal_repeats=config.temporal_repeats,
            num_condition_tokens=config.num_condition_tokens,
            cross_attention_heads=config.cross_attention_heads,
            r=config.r,
            bias=config.bias,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            modules_to_save=config.modules_to_save,
        )

    @classmethod
    def load_from_checkpoint(cls, *args, params=None, **kwargs: Any):
        return super().load_from_checkpoint(*args, params=params, **kwargs)


class TopDownPerceiver(LitPerceiverIO):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = TopDownPerceiverClassifier(
            ConditionedPerceiverConfig(
                backbone_model_name=self.hparams.backbone_model_name,
                num_backbone_channels=self.hparams.num_backbone_channels,
                batch_mixing=self.hparams.batch_mixing,
                temporal_repeats=self.hparams.temporal_repeats,
                num_condition_tokens=self.hparams.num_condition_tokens,
                cross_attention_heads=self.hparams.cross_attention_heads,
                r=self.hparams.r,
                bias=self.hparams.bias,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=self.hparams.target_modules,
                modules_to_save=self.hparams.modules_to_save,
            )
        )
        self.loss = nn.CrossEntropyLoss()
        self.acc = tm.classification.accuracy.Accuracy(task="multiclass", num_classes=1000)

    def forward(self, x, repeats=1, prev_state=None, conditioning=None):
        if x.ndim == 5:
            return self.model.view_movie(
                x, prev_state=prev_state, conditioning=conditioning
            )
        elif x.ndim == 4:
            return self.model.view_static_video(
                x, T=repeats, prev_state=prev_state, conditioning=conditioning
            )

    def loss_acc(self, logits, y):
        flat_y = y.flatten()
        loss = self.loss(logits.flatten(0, -2), flat_y)
        y_pred = logits.argmax(dim=-1)
        acc = self.acc(y_pred.flatten(), flat_y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        x, binarized_x, y = batch["pixel_values"], batch["binarized_pixel_values"], batch["label"]

        # Repeat the batch, binarize the last half
        x = torch.stack([x, binarized_x], dim=1)
        y = y.unsqueeze(1).repeat(1, 2)

        # Repeat the batch
        x = x.repeat(1, self.hparams.temporal_repeats, 1, 1, 1)
        y = y.repeat(1, self.hparams.temporal_repeats)

        # Iterate through
        x_lst = []
        y_lst = []
        for _ in range(self.hparams.batch_mixing):
            # Get a unique permutation of batches
            perm = torch.randperm(x.shape[0])
            x_lst.append(x[perm])
            y_lst.append(y[perm])

        # Concatenate
        x = torch.cat(x_lst, dim=1)
        y = torch.cat(y_lst, dim=1)

        # Get size
        B, T, C, H, W = x.shape

        # Generate random permutation indices for each sample in the batch
        perm_indices = torch.stack([torch.randperm(T, device=x.device) for _ in range(B)])

        # Expand the dimensions to match data shape
        expanded_indices = perm_indices.reshape(B, T, 1, 1, 1).expand(-1, -1, C, H, W)

        # Use gather to reorder the T dimension
        permuted_x = torch.gather(x, 1, expanded_indices)
        permuted_y = torch.gather(y, 1, perm_indices)

        outputs = self(permuted_x)
        logits = torch.stack([out.logits for out in outputs], dim=1)
        loss, acc = self.loss_acc(logits, permuted_y)

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, binarized_x, y = batch["pixel_values"], batch["binarized_pixel_values"], batch["label"]

        # Just concatenate the pre, gray, and post
        x = torch.stack([binarized_x, x, binarized_x], dim=1)

        # Get the outputs
        outputs = self(x)
        logits = torch.stack([out.logits for out in outputs], dim=1)

        _, pre_acc = self.loss_acc(logits[:, 0], y)
        self.log("val_pre_acc", pre_acc, sync_dist=True)

        _, gray_acc = self.loss_acc(logits[:, 1], y)
        self.log("val_gray_acc", gray_acc, sync_dist=True)

        _, post_acc = self.loss_acc(logits[:, 2], y)
        self.log("val_post_acc", post_acc, sync_dist=True)