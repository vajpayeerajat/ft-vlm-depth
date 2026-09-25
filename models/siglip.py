import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class SiglipVisionEncoder(nn.Module):
    """SigLIP Vision Encoder module for extracting visual token representations."""

    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384"):
        super().__init__()
        # Load pre-trained SigLIP model
        full_siglip = AutoModel.from_pretrained(model_id)
        
        # Keep only the vision tower
        self.vision_tower = full_siglip.vision_model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.hidden_size = self.vision_tower.config.hidden_size

        # Freeze vision encoder parameters by default
        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args:

            pixel_values: Image tensor [batch_size, channels, height, width]

        Returns:
            Image patch embeddings [batch_size, num_patches, vision_hidden_dim]
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        # Extract last hidden state tokens: [batch_size, num_patches, hidden_dim]
        image_embeds = vision_outputs.last_hidden_state
        return image_embeds