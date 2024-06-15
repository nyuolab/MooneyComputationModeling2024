import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from peft import LoraConfig, get_peft_model

from src.model.backbones.dinov2 import ConditionedDinov2

from dataclasses import dataclass


@dataclass
class ConditionedPerceiverConfig:
    backbone_model_name: str
    num_backbone_channels: int

    batch_mixing: int
    temporal_repeats: int
    num_condition_tokens: int

    cross_attention_heads: int

    r: int
    bias: str
    lora_alpha: float
    lora_dropout: float
    target_modules: list[str]
    modules_to_save: list[str]

    diagonal_attention: bool = False
    diagonal_attention_out: bool = False


@dataclass
class PerceiverOutput:
    logits: torch.Tensor
    query: torch.Tensor
    state: torch.Tensor
    cls: torch.Tensor
    latent_cls: torch.Tensor
    vis: torch.Tensor
    latent_vis: torch.Tensor
    conditioning: torch.Tensor
    latent_conditioning: torch.Tensor
    last_hidden: torch.Tensor
    hidden_states: list[torch.Tensor]
    attentions: list[torch.Tensor]


class TopDownPerceiverClassifier(nn.Module):
    def __init__(self, config: ConditionedPerceiverConfig):
        super().__init__()
        self.config = config

        # Tokens -1 must be square number
        num_query_side_length = int((config.num_condition_tokens - 1)**0.5)
        assert num_query_side_length ** 2 == config.num_condition_tokens - 1

        # Setup source model
        source_model = ConditionedDinov2.from_pretrained(config.backbone_model_name)
        lora_config = LoraConfig(
            r=config.r,
            bias=config.bias,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            modules_to_save=config.modules_to_save,
        )
        self.source_model = get_peft_model(source_model, lora_config)

        # Size of the unconditioned model
        with torch.no_grad():
            self.n_tokens = self.run_source(torch.zeros(1, 3, 224, 224)).hidden_states[-1].shape[1]

        # Downsampler for querying state
        hw = int((self.n_tokens-1)**0.5)
        self.downsampler = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=hw, w=hw),
            nn.AdaptiveAvgPool2d(num_query_side_length),
            Rearrange("b c h w -> b (h w) c"),
        )

        # Ratio of accumulation
        self.accumulate_state = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.num_condition_tokens = config.num_condition_tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=config.num_backbone_channels,
            num_heads=config.cross_attention_heads,
            batch_first=True,
        )
        self.kv_norm = nn.LayerNorm(config.num_backbone_channels)
        self.q_norm = nn.LayerNorm(config.num_backbone_channels)

        # Force attention parameter to be diagonal
        if config.diagonal_attention:
            self.attn.in_proj_weight = nn.Parameter(torch.cat([torch.eye(config.num_backbone_channels)] * 3, dim=0), requires_grad=False)
            self.attn.in_proj_bias = nn.Parameter(torch.cat([torch.ones(config.num_backbone_channels)] * 3, dim=0), requires_grad=False)
        if config.diagonal_attention_out:
            self.attn.out_proj.weight = nn.Parameter(torch.eye(config.num_backbone_channels), requires_grad=False)
            self.attn.out_proj.bias = nn.Parameter(torch.zeros(config.num_backbone_channels), requires_grad=False)

        # Initial state
        self.init_state = nn.Parameter(torch.randn(1, self.n_tokens+self.num_condition_tokens, config.num_backbone_channels))

    def run_source(self, x, conditioning=None):
        return self.source_model(x, return_dict=True, output_hidden_states=True, output_attentions=True, conditioning=conditioning)

    def forward(self, x, prev_state=None, conditioning=None):
        # None conditioning if no previous latents provided
        if prev_state is None:
            prev_state = self.init_state.repeat(x.shape[0], 1, 1)

        # Query tokens are generated
        unconditioned_output = self.run_source(x, conditioning=None)
        unconditioned_cls = unconditioned_output.hidden_states[-1][:, 0:1]
        unconditioned_vis = unconditioned_output.hidden_states[-1][:, 1:]
        query_tokens = torch.cat([unconditioned_cls, self.downsampler(unconditioned_vis)], dim=1)

        # Query state and obtain conditioning
        normalized_query_tokens = self.q_norm(query_tokens)
        normalized_state = self.kv_norm(prev_state)
        conds = self.attn(
            query=normalized_query_tokens,
            key=normalized_state,
            value=normalized_state
        )[0] + normalized_query_tokens

        # Put conditioning through the source model
        output = self.run_source(x, conditioning=conds)
        logits = output.logits
        cls = output.hidden_states[-1][:, 0:1]
        vis = output.hidden_states[-1][:, 1+self.num_condition_tokens:]

        # Update state
        new_state = prev_state * self.accumulate_state + output.hidden_states[-1] * (1 - self.accumulate_state)

        return PerceiverOutput(
            logits=logits,
            cls=cls,
            latent_cls=new_state[:, :1],
            vis=vis,
            latent_vis=new_state[:, 1+self.num_condition_tokens:],
            query=query_tokens,
            state=new_state,
            conditioning=conds,
            latent_conditioning=conds,
            last_hidden=output.hidden_states[-1],
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def view_movie(
        self,
        x: torch.Tensor,
        prev_state: torch.Tensor = None,
        conditioning: torch.Tensor = None,
    ):
        # [B, T, C, H, W]
        outputs = []
        for t in range(x.shape[1]):
            step_out = self(
                x[:, t],
                prev_state=prev_state,
                conditioning=conditioning if t == 0 else None
            )

            # Update previous latents
            prev_state = step_out.state

            # Append to outputs
            outputs.append(step_out)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def view_static_video(
        self, x, T, 
        prev_state: torch.Tensor = None,
        conditioning=None
    ):
        x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        return self.view_movie(x,
            prev_state=prev_state,
            conditioning=conditioning
        )
