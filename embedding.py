import torch
import torch.nn as nn

# An Embedding layer is basically a lookup table: each token id gets its own
# learnable vector. So if it's Embedding(5, 3), we have 5 tokens in our
# vocabulary and each one is represented by a vector of 3 numbers.
# Unlike Linear, there's no multiplication here, we just grab rows.
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        # shape is (vocab_size, d_model): one row per token, so row i IS the
        # vector for token id i. (Different from Linear, where rows are outputs.)
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model, device=device, dtype=dtype))

    def forward(self, ids):
        # ids is (batch, seq_len) of integer token ids. Indexing the weight
        # with them swaps each id for its row, adding a d_model dimension:
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.weight[ids]


if __name__ == "__main__":
    # The idea: turn words into vectors the model can do math with.
    #  Vocabulary (5 tokens): 0="the", 1="cat", 2="sat", 3="on", 4="mat"
    #  Each word becomes a vector of 3 numbers
    emb = Embedding(vocab_size=5, d_model=3)

    # A batch with 1 sentence: "the sat sat" -> shape (1, 3)
    # (integers, because these are ids we index with, not values we multiply)
    sentence_batch = torch.tensor([[0, 2, 2]])

    vectors = emb(sentence_batch)
    print("vectors:", vectors)

    # The same word always maps to the same vector, wherever it appears.

    # The table is untrained (random weights), so these vectors are
    # meaningless for now. Training is what would push similar words
    # (like "cat" and "dog") to have similar vectors.
    print("Shape (batch, seq_len, d_model):", vectors.shape)  # torch.Size([1, 3, 3])