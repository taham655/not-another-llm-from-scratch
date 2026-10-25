import pickle

import torch

from bpe import Tokenizer
from model import GPT
from train import DEVICE, MODEL
from utils import load_checkpoint, softmax


# Repeatedly: take the last row of logits, sample a token, append it.
# temperature < 1 = safer/more repetitive, > 1 = more random.
# top_p: only sample from the most likely tokens that add up to top_p.
@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=0.8, top_p=0.9, eot_id=None):
    for _ in range(max_new_tokens):
        logits = model(ids[:, -MODEL["context_length"]:])[:, -1, :]
        probs = softmax(logits / temperature)
        sorted_probs, order = probs.sort(descending=True)
        sorted_probs[sorted_probs.cumsum(-1) - sorted_probs >= top_p] = 0
        next_id = order.gather(-1, torch.multinomial(sorted_probs, 1))
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == eot_id:
            break
    return ids


if __name__ == "__main__":
    model = GPT(**MODEL)
    load_checkpoint("checkpoint.pt", model)
    model.to(DEVICE)
    tok = Tokenizer(**pickle.load(open("data/tokenizer.pkl", "rb")))

    ids = torch.tensor([tok.encode("Once upon a time")], device=DEVICE)
    out = generate(model, ids, 256, eot_id=tok.id_of[b"<|endoftext|>"])
    print(tok.decode(out[0].tolist()))
