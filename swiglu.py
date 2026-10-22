import torch
import torch.nn as nn

from linear import Linear

# SiLU (a.k.a. Swish) is a smooth version of ReLU: x * sigmoid(x).
# Big positive x -> sigmoid is ~1, so it passes x through almost unchanged.
# Big negative x -> sigmoid is ~0, so it squashes x to ~0.
# Unlike ReLU it lets a little bit of negative values through, and has no sharp corner at 0.
def silu(x):
    return x * torch.sigmoid(x)


# SwiGLU is the feed-forward block used in LLaMA-style transformers.
# A plain feed-forward block is: Linear -> activation -> Linear.
# SwiGLU adds a "gate": we make TWO projections of the input, run one through
# SiLU, and multiply them together element-wise. The SiLU side decides how much
# of the other side gets through (like a dimmer switch per feature).
#   SwiGLU(x) = W2( silu(W1 x) * (W3 x) )
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # w1 and w3 both go up from d_model to the bigger hidden size d_ff,
        # w2 brings it back down so the output has the same shape as the input.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down projection

    def forward(self, x):
        # x is (batch, seq_len, d_model)
        gate = silu(self.w1(x))   # (batch, seq_len, d_ff)
        value = self.w3(x)        # (batch, seq_len, d_ff)
        # element-wise multiply, then project back:
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.w2(gate * value)


if __name__ == "__main__":
    # First, what SiLU does to a few numbers:
    nums = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    print("x:      ", nums)
    print("silu(x):", silu(nums))  # ~[-0.03, -0.27, 0, 0.73, 4.97]

    # The idea: take the vector for each token (like what Embedding gives us)
    # and "think about it" in a bigger space before coming back to d_model.
    # d_ff is usually ~8/3 * d_model in LLaMA; here 4 -> 8 just to keep it small.
    ffn = SwiGLU(d_model=4, d_ff=8)

    # A batch with 1 sentence of 3 tokens, each a vector of 4 numbers
    # -> shape (1, 3, 4)
    token_vectors = torch.randn(1, 3, 4)

    out = ffn(token_vectors)
    print("out:", out)

    # Each token is processed on its own (tokens don't talk to each other here,
    # that's attention's job), and the shape comes out the same as it went in.
    # Untrained (random weights), so the numbers are meaningless for now.
    print("Shape (batch, seq_len, d_model):", out.shape)  # torch.Size([1, 3, 4])
