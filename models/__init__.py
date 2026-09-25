import torch
import torch.nn as nn

from .gemma import GemmaDecoder
from .siglip import SiglipVisionEncoder


class PaliGemmaModel(nn.Module):
    """PaliGemma Multi-modal Architecture combining SigLIP + Projector + Gemma."""

    def __init__(
        self,
        siglip_id: str = "google/siglip-so400m-patch14-384",
        gemma_id: str = "google/gemma-2b",
    ):
        super().__init__()
        self.vision_encoder = SiglipVisionEncoder(siglip_id)
        self.decoder = GemmaDecoder(gemma_id)

        # Connect vision encoder hidden size to gemma decoder hidden size
        self.projector = nn.Linear(
            self.vision_encoder.hidden_size, self.decoder.hidden_size
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
    ):
        # Extract raw patch tokens from SigLIP
        vision_features = self.vision_encoder(pixel_values)

        # Project vision dimensions to match Gemma language embedding dimension
        projected_vision_embeds = self.projector(vision_features)

        # Forward pass through Gemma
        loss, logits = self.decoder(
            input_ids=input_ids,
            vision_projected_embeds=projected_vision_embeds,
            labels=labels,
            attention_mask=attention_mask,
        )

        return loss, logits


__all__ = [
    "SiglipVisionEncoder",
    "PaliGemmaMultiModalProjector",
    "GemmaDecoder",
    "PaliGemmaModel",
]