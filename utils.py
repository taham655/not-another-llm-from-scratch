import math

import numpy as np
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


# Learning rate schedule: how big a step to take at training step t.
#   1. warmup: ramp up in a straight line from 0 to max_lr. Weights are random
#      and Adam's v is still empty, so full-size steps early can blow up.
#   2. cosine decay: follow half a cosine from max_lr down to min_lr, so steps
#      shrink as the model gets close to a good solution.
#   3. after that: stay at min_lr.
# cosine_cycle_iters is the step where the decay ENDS, not how long it lasts.
def get_lr_cosine_schedule(t, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        return t / warmup_iters * max_lr
    if t > cosine_cycle_iters:
        return min_lr
    # how far through the decay we are: 0 at the start, 1 at the end
    progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    # cos goes 1 -> -1, so (1 + cos) / 2 goes 1 -> 0
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


# Gradient clipping: a speed limit on the gradient, so one bad batch with a
# huge gradient can't throw the weights far away.
#   1. measure ALL the model's gradients as one long vector, using its
#      L2 norm (length): sqrt(sum of every grad number squared)
#   2. if that length is over max_norm, multiply every grad by
#      max_norm / norm, so the new length is exactly max_norm
# One shared factor keeps the direction the same and only shrinks the
# length. Call it between loss.backward() and opt.step().
def gradient_clipping(parameters, max_norm, eps=1e-6):
    # parameters is often a generator (model.parameters()), which can only be
    # looped over once, and we need two passes: measure, then shrink
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = math.sqrt(sum((g ** 2).sum().item() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)  # in place, so opt.step() sees the clipped grads
    return total_norm  # pre-clip norm, worth logging during training


# Data loading: the whole dataset is one long array of token ids. A batch is
# batch_size random windows of context_length tokens; y is x shifted by one,
# so position i is graded on the token right after it.
def get_batch(data, batch_size, context_length, device):
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[s : s + context_length] for s in starts])
    y = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    # uint16 on disk -> int64 for indexing the embedding table
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device))


# Checkpointing: everything needed to resume exactly where we stopped. The
# optimizer state matters too: AdamW's m, v and t live there.
def save_checkpoint(model, optimizer, iteration, out):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer=None):
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


if __name__ == "__main__":
    # Schedule used by the official test: warmup to step 7, decay until step 21.
    lrs = [get_lr_cosine_schedule(t, 1.0, 0.1, 7, 21) for t in range(25)]
    assert lrs[0] == 0 and lrs[7] == 1.0  # ramp starts at 0, peaks after warmup
    assert abs(lrs[14] - 0.55) < 1e-9     # halfway through decay = midpoint of 1 and 0.1
    assert lrs[21] == lrs[24] == 0.1      # decay ends at min_lr and stays there
    print("lr schedule:", [round(lr, 3) for lr in lrs])

    # Two "weights" with grads [3, 4] and [12]: global norm = sqrt(9+16+144) = 13.
    a = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    a.grad, b.grad = torch.tensor([3.0, 4.0]), torch.tensor([12.0])
    gradient_clipping([a, b], max_norm=1.0)
    # every grad was multiplied by ~1/13: same 3:4:12 ratio, new length 1
    assert torch.allclose(torch.cat([a.grad, b.grad]), torch.tensor([3.0, 4.0, 12.0]) / 13, atol=1e-5)
    gradient_clipping([a, b], max_norm=5.0)  # already under the limit: untouched
    assert torch.allclose(torch.cat([a.grad, b.grad]), torch.tensor([3.0, 4.0, 12.0]) / 13, atol=1e-5)
    print("clipped grads:", a.grad, b.grad)

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
