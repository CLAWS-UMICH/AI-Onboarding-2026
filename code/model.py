"""The intent model. Write your own version in your folder.

Outline (do it however you like):
- A pretrained encoder from Hugging Face (MiniLM, BGE-small, MPNet; see MODELS.md).
- Pool the token outputs into one vector per sentence (mean pooling works well).
- A classification head that maps that vector to one score per intent.
- Use MPS/CUDA if available, otherwise CPU.
"""
