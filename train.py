import time

import numpy as np
import torch
import trackio

from adamw import Adamw
from model import GPT
from utils import cross_entropy_loss, get_batch, get_lr_cosine_schedule, gradient_clipping, save_checkpoint

MODEL = dict(vocab_size=10_000, context_length=256, d_model=512, num_layers=4,
             num_heads=16, d_ff=1344, rope_theta=10_000.0)
BATCH_SIZE = 32
STEPS = 5000  # 32 * 5000 * 256 = ~41M tokens
MAX_LR, MIN_LR, WARMUP = 1e-3, 1e-4, 200
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
# M3 Max 30-core GPU, float32: 30 cores * 128 ALUs * 2 (multiply-add) * ~1.38 GHz
PEAK_FLOPS = 10.6e12

if __name__ == "__main__":
    train_data = np.load("data/train.npy", mmap_mode="r")
    valid_data = np.load("data/valid.npy", mmap_mode="r")

    model = GPT(**MODEL).to(DEVICE)
    opt = Adamw(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)

    # FLOPs per token for forward + backward: 6 per weight that's used in a matmul
    # (the embedding table is a lookup, so it's left out), plus attention's
    # QK^T and weights@V, which don't involve weights at all.
    n_matmul_params = sum(p.numel() for p in model.parameters()) - model.token_embedding.weight.numel()
    L, d, T = MODEL["num_layers"], MODEL["d_model"], MODEL["context_length"]
    flops_per_token = 6 * n_matmul_params + 12 * L * T * d

    trackio.init(project="not-another-llm-from-scratch", config={
        **MODEL, "batch_size": BATCH_SIZE, "steps": STEPS, "max_lr": MAX_LR,
        "min_lr": MIN_LR, "warmup": WARMUP, "device": DEVICE,
    })

    t0 = time.time()
    for step in range(STEPS):
        lr = get_lr_cosine_schedule(step, MAX_LR, MIN_LR, WARMUP, STEPS)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, MODEL["context_length"], DEVICE)
        loss = cross_entropy_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            # MFU = FLOPs we actually did per second / the GPU's peak FLOPs per second
            # (loss.item() waits for the GPU to finish, so the timing is honest)
            train_loss = loss.item()
            steps_done = 100 if step > 0 else 1
            tokens_per_sec = steps_done * BATCH_SIZE * T / (time.time() - t0)
            mfu = flops_per_token * tokens_per_sec / PEAK_FLOPS
            with torch.no_grad():
                x, y = get_batch(valid_data, BATCH_SIZE, T, DEVICE)
                val_loss = cross_entropy_loss(model(x), y).item()
            print(f"step {step} | loss {train_loss:.3f} | val {val_loss:.3f} | lr {lr:.1e} "
                  f"| {tokens_per_sec:,.0f} tok/s | MFU {mfu:.1%}")
            trackio.log({"train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                        "tokens_per_sec": tokens_per_sec}, step=step)
            t0 = time.time()

    save_checkpoint(model, opt, STEPS, "checkpoint.pt")
    trackio.finish()
