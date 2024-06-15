import torch
from typing import Optional

from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings, Dinov2Model, Dinov2ForImageClassification, BaseModelOutputWithPooling, ImageClassifierOutput


class ConditionedDinov2Embeddings(Dinov2Embeddings):
    def forward(self, pixel_values: torch.Tensor, conditioning: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = super().forward(pixel_values, bool_masked_pos=bool_masked_pos)
        if conditioning is None:
            return embeddings
        else:
            return torch.cat([embeddings[:, :1], conditioning, embeddings[:, 1:]], dim=1)


class ConditionedDinov2Model(Dinov2Model):
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, conditioning=conditioning)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ConditionedDinov2(Dinov2ForImageClassification):
    def post_init(self):
        # Replace the embeddings with the conditioned embeddings
        self.dinov2 = ConditionedDinov2Model(self.config)
        self.dinov2.embeddings = ConditionedDinov2Embeddings(self.config)
        super().post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.dinov2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            conditioning=conditioning,
        )

        sequence_output = outputs[0]

        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]

        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        logits = self.classifier(linear_input)

        loss = None
        assert labels is None
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )