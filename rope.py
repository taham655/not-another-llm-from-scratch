import torch
import torch.nn as nn
from einops import rearrange

# RoPE (Rotary Positional Embedding) tells the model WHERE each token is.
# Attention on its own has no idea about word order: "cat sat" and "sat cat"
# look the same. RoPE fixes that by rotating each token's vector by an angle
# that depends on its position.
#
# How: split the vector into pairs of numbers, (x0, x1), (x2, x3), ...
# and treat each pair as a 2D point. For a token at position m, rotate pair i
# by the angle m * freq_i. Early pairs spin fast, later pairs spin slowly
# (like the hands of a clock: seconds, minutes, hours).
#
# The nice part: when attention takes the dot product of two rotated vectors,
# only the DIFFERENCE between their angles survives, so the model sees
# relative distance ("3 tokens apart") rather than absolute position.
# There are no learnable weights here, it's pure geometry.
class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        # one frequency per pair: theta^(-0/d_k), theta^(-2/d_k), ... -> (d_k/2,)
        # theta is usually 10000, which makes the last pairs rotate very slowly.
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        # angle for every (position, pair): (max_seq_len, d_k/2)
        angles = torch.outer(positions, freqs)
        # precompute cos/sin once. register_buffer (not nn.Parameter) because
        # these aren't learned, but should still move with .to(device).
        # persistent=False keeps them out of the saved state_dict.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        # x is (..., seq_len, d_k), token_positions is (..., seq_len) of ints.
        # Look up the cos/sin for each token's position: (..., seq_len, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # group the numbers into pairs: (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
        # then pull out the two halves of each pair: (..., seq_len, d_k/2) each
        pairs = rearrange(x, "... (half two) -> ... half two", two=2)
        x_even, x_odd = pairs[..., 0], pairs[..., 1]

        # standard 2D rotation of the point (x_even, x_odd) by the angle
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # zip the pairs back together in their original order:
        # (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
        rotated = torch.stack([rot_even, rot_odd], dim=-1)
        return rearrange(rotated, "... half two -> ... (half two)")


if __name__ == "__main__":
    rope = RoPE(theta=10000.0, d_k=4, max_seq_len=16)

    # The idea: the same word at different positions should look different,
    # so the model can tell them apart.
    # Take one "query" vector q and one "key" vector k (like in attention).
    q = torch.randn(4)
    k = torch.randn(4)

    def score(q_pos, k_pos):
        # rotate q as if it were at q_pos and k as if it were at k_pos,
        # then take the dot product (that's what attention computes)
        # (d_k,) -> (1, d_k): a "sentence" of 1 token
        q_rot = rope(rearrange(q, "d -> 1 d"), torch.tensor([q_pos]))
        k_rot = rope(rearrange(k, "d -> 1 d"), torch.tensor([k_pos]))
        return (q_rot @ k_rot.T).item()

    # Same distance apart (3 tokens) -> same score, wherever they are.
    print("score q@2, k@5:  ", score(2, 5))
    print("score q@10, k@13:", score(10, 13))
    # Different distance -> different score.
    print("score q@2, k@9:  ", score(2, 9))

    # Rotation only changes direction, never length, so the vector's
    # size (norm) is the same before and after.
    rotated = rope(rearrange(q, "d -> 1 d"), torch.tensor([7]))
    print("norm before:", q.norm().item(), " after:", rotated.norm().item())

    # A batch with 1 sentence of 3 tokens, each a vector of 4 -> (1, 3, 4)
    x = torch.randn(1, 3, 4)
    positions = rearrange(torch.arange(3), "s -> 1 s")  # [[0, 1, 2]]
    out = rope(x, positions)
    # Position 0 has angle 0, so the first token is left unchanged.
    print("token 0 unchanged:", torch.allclose(out[0, 0], x[0, 0]))
    print("Shape (batch, seq_len, d_k):", out.shape)  # torch.Size([1, 3, 4])
