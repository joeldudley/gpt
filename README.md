# Simple GPT

An implementation of the GPT-2 architecture with PyTorch. It implements its own attention blocks, while using
off-the-shelf PyTorch components for neural networks/embeddings/optimizers/etc.

The focus is on simplicity over performance. With this in mind, it's made various sacrifices (no GPU support, no
obtuse performance optimisations, no ability to save trained model weights...).

The tests under `tests/test.py` show the GPT in action, learning to sum two-digit numbers with 100% accuracy.
