"""
DIRECTIONS -------------------------------------------------------

The intent model. Write your own version in your folder.

Outline (do it however you like):
- A pretrained encoder from Hugging Face (MiniLM, BGE-small, MPNet; see MODELS.md).
- Pool the token outputs into one vector per sentence (mean pooling works well).
- A classification head that maps that vector to one score per intent.
- Use MPS/CUDA if available, otherwise CPU.

STEPS -----------------------------------------------------------

1. Load the pretrained encoder named by INTENT_MODEL (default MiniLM), weights are trainable
2. Encoder gives one vector per token; mean pool them (ignoring padding) into one sentence vector
3. Dropout + one linear layer maps the sentence vector (384 or 768 numbers) to one score per intent
4. Scores are raw logits; train.py applies cross-entropy, predict.py applies softmax
5. pick_device() chooses MPS (Mac GPU), then CUDA (Nvidia GPU), then CPU
"""

# LIBRARIES ------------------------------------------------------

import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("INTENT_MODEL", "sentence-transformers/all-MiniLM-L6-v2") # default is MiniLM
MAX_TOKENS = 64 # training sentences are 2-20 words, so 64 tokens is enuff

# Note: baseline.py, train.py and predict.py should all use this same MODEL_NAME

# HELPER FUNCTIONS ------------------------------------------------

# Goal: Turn token vectors into one sentence vector by averaging, but skipping padding tokens
# Padding tokens: dummy tokens added to make all sentences in a batch the same length, needed because neural networks need rectangular inputs (i.e. same tokens per sentence)
def mean_pool(token_vecs, attention_mask):
    # attention_mask is shape (batch, tokens) with 1s for real tokens, 0s for padding
    # unsqueeze(-1) adds a dimension: (batch, tokens, 1)
    # .to(token_vecs.dtype) converts it to float so you can multiply with vectors
    # Result: a mask you can multiply with to zero out padding
    mask = attention_mask.unsqueeze(-1).to(token_vecs.dtype)

    # token_vecs * mask multiplies each token vector by its mask (0 if padding, 1 if real)
    # .sum() adds up all real token vecotrs, ignoring padding
    # Result: (batch, hidden) = one summed up vector per sentence
    summed = (token_vecs * mask).sum(dim=1)

    # mask.sum(dim=1) counts how many real tokens each sentence has
    # .clamp() prevents division by zero (if sentence somehow has no tokens)
    # Result: (batch, 1) = total number of real tokens per sentence
    counts = mask.sum(dim=1).clamp(min=1e-9)

    # Result: (batch, hidden) = one averaged vector per sentence (padding ignored)
    return summed / counts


# Goal: Build a model that takes sentences and outputs intent scores (logits)
class IntentModel(nn.Module):
    def __init__(self, num_intents, model_name=MODEL_NAME, dropout=0.1):
        # initializes parent class - standard PyTorch
        super().__init__()

        # Loads pretrained encoder from Hugging Face - these weights are trainable
        self.encoder = AutoModel.from_pretrained(model_name)

        # Reads encoder's output size (384 for MiniLM, 768 for MPNet) - need this to bulid next layer
        hidden = self.encoder.config.hidden_size

        # Adds random dropout (10% of neurons turned off during training) - helps prevent overfitting
        # Only active when model.train() is called, turned off in model.eval()
        self.dropout = nn.Dropout(dropout)

        # Single linear layer that maps the 384 sentence vector down to 36 intent scores
        self.head = nn.Linear(hidden, num_intents)

    # Goal: Forward pass, called when you run the model
    def forward(self, batch):
        # Runs encoder on the batch and returns token vectors, shape (batch, tokens, hidden)
        token_vecs = self.encoder(**batch).last_hidden_state

        # Pools the tokens into one vector per sentence, shape (batch,hidden). Skipps padding
        sentence_vecs = mean_pool(token_vecs, batch["attention_mask"])

        # Applies dropout to the sentence vectors, then the linear layer, returns 36 scores per sentence (logits, not probabilities yet)
        return self.head(self.dropout(sentence_vecs))


# Goal: load tokenizer that matches the encoder so sentences are split the same way the encoder was trained on
# Why: Each encoder was trained on text split a specific way ("test-case" vs "test" + "case")
def load_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)

# Goal: Turn a list of sentences into PyTorch tensors on right device
def tokenize(tokenizer, texts, device):
    # Tokenizes sentences
    # padding = true - pads short sentences to longest one in batch, truncation = True - cuts long sentences to 64 tokens (MAX_TOKENS), return_tensors = "pt" - return PyTorch tensors, not lists
    # Result: dict with "input_ids" (token numbers), "attention_mask" (real vs. padding), etc.
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")

    # Loops through dict and moves each tensor to the device
    # .to(device) puts it where the model lives, so they can compute together
    # Result: same dict, but all tensors are on the right device and ready for model
    return {k: v.to(device) for k, v in enc.items()}


# Fastest available device: Mac GPU, then Nvidia GPU, then CPU
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
