"""Intent model: pretrained encoder -> mean pooling -> dropout -> linear head (one score per intent)."""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 64  # training sentences are at most ~20 words


class IntentModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean pooling: average token vectors, ignoring padding.
        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(self.dropout(pooled))


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def encode(tokenizer, texts, dev):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return batch["input_ids"].to(dev), batch["attention_mask"].to(dev)


@torch.no_grad()
def predict_proba(model, tokenizer, texts, dev, batch_size=128):
    """Softmax probabilities, shape (len(texts), num_labels)."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        ids, mask = encode(tokenizer, texts[i:i + batch_size], dev)
        out.append(torch.softmax(model(ids, mask), dim=-1).cpu())
    return torch.cat(out)
