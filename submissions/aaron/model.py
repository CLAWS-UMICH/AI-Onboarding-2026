"""The intent model: MPNet (with its mean pooling) + one linear layer -> 36 scores."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from common import LABELS

MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_LEN = 32  # tokens; commands are short, so this only speeds things up
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class IntentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SentenceTransformer = MPNet + mean pooling + normalize, one 768-number vector per sentence.
        self.encoder = SentenceTransformer(MODEL, device=DEVICE)
        self.encoder.max_seq_length = MAX_LEN
        self.head = nn.Linear(
            self.encoder.get_embedding_dimension(), len(LABELS)
        )

    def forward(self, texts):
        """list of sentences -> logits (raw scores), shape [batch, 36]."""
        # preprocess() also returns non-tensor entries; batch_to_device only moves the tensors.
        features = batch_to_device(self.encoder.preprocess(texts), DEVICE)
        return self.head(self.encoder(features)["sentence_embedding"])
