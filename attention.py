import torch
import torch.nn as nn
from einops import rearrange

from linear import Linear
from rope import RoPE
from utils import scaled_dot_product_attention

# Multi-head self-attention: this is where tokens finally talk to each other.
# Every token makes a query, key and value (via Linear layers), and then
# scaled dot-product attention mixes information between tokens.
#
# "Multi-head": instead of one big attention over d_model numbers, we split
# the vector into num_heads smaller chunks and run attention on each chunk
# separately. Each head can learn to look for something different
# (e.g. one tracks the previous word, another tracks the subject).
# Then we glue the heads back together and mix them with one more Linear.
#
# "Self": queries, keys and values all come from the same sentence.
# "Causal": each token can only look at itself and earlier tokens, since
# a language model predicting the next word shouldn't peek at it.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must split evenly across heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # size of each head's chunk
        # one Linear each for queries, keys, values (covers all heads at once),
        # and one to mix the heads back together at the end
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # optional RoPE (from rope.py) so attention knows token positions.
        # It has no weights, so one RoPE can be shared by every layer.
        self.rope = rope

    def split_heads(self, x):
        # cut d_model into num_heads chunks of d_k and move heads in front of
        # seq_len, so each head is its own "batch":
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        return rearrange(x, "b s (h d) -> b h s d", h=self.num_heads)

    def forward(self, x, token_positions=None):
        # x is (batch, seq_len, d_model)
        seq_len = x.shape[1]

        # (batch, num_heads, seq_len, d_k) each
        Q = self.split_heads(self.q_proj(x))
        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)  # [0, 1, ..., seq_len-1]
            # rotate queries and keys (not values: position only matters for
            # deciding WHO to look at, not for WHAT gets passed along).
            # Every head uses the same positions, so RoPE is applied per head.
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # causal mask: (seq_len, seq_len), True on and below the diagonal.
        # It broadcasts over batch and heads.
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)  # (batch, num_heads, seq_len, d_k)

        # glue the heads back together (the reverse of split_heads):
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


if __name__ == "__main__":
    # d_model = 8 split into 2 heads of 4. RoPE works on each head's chunk,
    # so its d_k is the head size (4), not d_model.
    rope = RoPE(theta=10000.0, d_k=4, max_seq_len=16)
    attn = MultiHeadSelfAttention(d_model=8, num_heads=2, rope=rope)

    # A batch with 1 sentence of 3 tokens, each a vector of 8 -> (1, 3, 8)
    # (in a real model these would come from Embedding)
    x = torch.randn(1, 3, 8)
    out = attn(x)
    print("out:", out)
    print("Shape (batch, seq_len, d_model):", out.shape)  # torch.Size([1, 3, 8])

    # Checking the causal mask: change the LAST token and the earlier tokens'
    # outputs shouldn't move, because they're not allowed to look ahead.
    x2 = x.clone()
    x2[0, 2] = torch.randn(8)
    out2 = attn(x2)
    print("tokens 0-1 unchanged:", torch.allclose(out[0, :2], out2[0, :2]))  # True
    print("token 2 changed:     ", not torch.allclose(out[0, 2], out2[0, 2]))  # True
