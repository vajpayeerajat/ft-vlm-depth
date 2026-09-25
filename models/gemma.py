from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaDecoder(nn.Module):
    """Gemma Causal Language Model supporting multi-modal vision-token insertion."""

    def __init__(self, model_id: str = "google/gemma-2b"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.hidden_size = self.decoder.config.hidden_size

    def forward(self, input_ids: torch.LongTensor, vision_projected_embeds: torch.Tensor, labels: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            input_ids: Text token IDs [batch_size, seq_len]
            vision_projected_embeds: Projected vision tokens [batch_size,
              num_patches, hidden_size]
            labels: Target token IDs [batch_size, seq_len + num_patches]
            attention_mask: Mask for input sequence

        Returns:
            Tuple of (loss, logits)
        """
        # 1. Convert text input_ids into embedding space
        text_embeds = self.decoder.get_input_embeddings()(input_ids)

        # 2. Concatenate vision tokens before text tokens along sequence length (dim=1)
        #    Result shape: [batch_size, num_patches + seq_len, hidden_size]
        combined_embeds = torch.cat([vision_projected_embeds, text_embeds], dim=1)

        # 3. Construct/adjust attention mask if provided
        if attention_mask is not None:
            batch_size, num_patches = vision_projected_embeds.shape[:2]
            prefix_mask = torch.ones(
                (batch_size, num_patches),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 4. Pass merged embeddings directly into Gemma's forward pass
        outputs = self.decoder(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_attentions=False,
        )

        return outputs.loss, outputs.logits