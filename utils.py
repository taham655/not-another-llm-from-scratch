import math

import torch
from einops import einsum, rearrange

# Softmax turns a list of any numbers ("scores") into probabilities:
# all positive, and they add up to 1. Bigger score -> bigger share.
#   softmax(x)_i = exp(x_i) / sum_j exp(x_j)
# exp() blows up fast (exp(1000) is inf), so we first subtract the max.
# That doesn't change the answer (it cancels out top and bottom), but now
# the biggest number we exp() is 0, so nothing overflows.
def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x)
    return exps / exps.sum(dim=dim, keepdim=True)


# Scaled dot-product attention: each token asks a question (query), every
# token offers a label (key) and some content (value).
#   1. score = how well my query matches each key (dot product)
#   2. softmax the scores into weights that add up to 1
#   3. output = weighted average of the values
# We divide by sqrt(d_k) because dot products get bigger as vectors get
# longer, and huge scores make softmax put ~all the weight on one token.
#
# mask is True where attention is ALLOWED. Blocked spots get -inf, which
# softmax turns into exactly 0 weight.
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q is (..., seq_q, d_k), K is (..., seq_k, d_k), V is (..., seq_k, d_v)
    d_k = Q.shape[-1]
    # dot every query with every key (d is summed away):
    # (..., seq_q, d_k), (..., seq_k, d_k) -> (..., seq_q, seq_k)
    # row i = how much query i matches every key
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    # weighted sum of the values (k is summed away):
    # (..., seq_q, seq_k), (..., seq_k, d_v) -> (..., seq_q, d_v)
    return einsum(weights, V, "... q k, ... k d -> ... q d")


def cross_entropy_loss(logits, targets):
    max_logit = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logit
    exps = torch.exp(shifted)
    sum_exp = exps.sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    final = log_sum_exp + max_logit

    index = rearrange(targets, "... -> ... 1")

    correct = torch.gather(logits, dim=-1, index=index)

    per_pos = final - correct

    loss = per_pos.mean()

    return loss








if __name__ == "__main__":
    scores = torch.tensor([1.0, 2.0, 3.0])
    print("softmax:", softmax(scores))  # ~[0.09, 0.24, 0.67], sums to 1

    # Same gaps between numbers -> same probabilities, even when the numbers
    # are huge. Without subtracting the max this would be all nan.
    print("softmax (huge):", softmax(scores + 1000))

    # Attention on a sentence of 3 tokens, each with d_k = 2 -> (3, 2)
    Q = torch.randn(3, 2)
    K = torch.randn(3, 2)
    # make each value easy to spot: token 0 -> [1,0], 1 -> [0,1], 2 -> [1,1]
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Causal mask: token i can only look at tokens 0..i (not the future).
    #  [[T, F, F],
    #   [T, T, F],
    #   [T, T, T]]
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))

    out = scaled_dot_product_attention(Q, K, V, mask)
    # token 0 can only see itself, so its output is exactly V[0] = [1, 0]
    print("out:", out)
    print("Shape (seq_len, d_v):", out.shape)  # torch.Size([3, 2])
