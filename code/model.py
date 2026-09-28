"""MiniLM encoder + mean pooling + one linear layer (same shape as EVA/Models/singleintentmodel)."""
import torch
from torch import nn
from transformers import AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class IntentModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, batch):
        hidden = self.encoder(**batch).last_hidden_state   # (B, tokens, 384)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)                           # (B, num_labels) logits


def best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
