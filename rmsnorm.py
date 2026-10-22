import torch
import torch.nn as nn

# RMSNorm keeps each token's vector at a steady size as it flows through
# many layers. Without it, numbers can grow (or shrink) a little at every
# layer until they blow up (or vanish) by the end.
#
# How: measure the vector's "typical size" with its root-mean-square,
#   rms(x) = sqrt(mean(x^2))
# divide by it so the vector has rms 1, then multiply by a learnable gain
# (one number per feature) so the model can still make some features bigger.
#   RMSNorm(x) = x / rms(x) * gain
# It's a simpler LayerNorm: no subtracting the mean, no bias. LLaMA uses it.
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        # eps stops us dividing by 0 if a vector is all zeros
        self.eps = eps
        # starts at 1, so at first RMSNorm only rescales and doesn't reweight
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        # x is (batch, seq_len, d_model). Each token is normalised on its own,
        # over its d_model numbers, so the shape doesn't change.
        in_dtype = x.dtype
        # squaring small/large numbers in float16 can overflow, so do the
        # math in float32 and convert back at the end
        x = x.to(torch.float32)
        # (batch, seq_len, d_model) -> (batch, seq_len, 1): one rms per token
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


if __name__ == "__main__":
    norm = RMSNorm(d_model=4)

    # A batch with 1 sentence of 2 tokens, each a vector of 4 -> (1, 2, 4)
    # token 0 is normal-sized numbers, token 1 is huge numbers
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0],
                       [1000.0, -2000.0, 3000.0, -4000.0]]])

    out = norm(x)
    print("out:", out)

    # Both tokens come out the same: the direction is kept, the size is not.
    # Token 1 is just token 0 times 1000, so after normalising they match.
    print("same after norm:", torch.allclose(out[0, 0], out[0, 1]))  # True

    # Every token now has rms ~1, no matter how big it was going in.
    print("rms per token:", out.pow(2).mean(dim=-1).sqrt())  # ~[[1., 1.]]
    print("Shape (batch, seq_len, d_model):", out.shape)  # torch.Size([1, 2, 4])
