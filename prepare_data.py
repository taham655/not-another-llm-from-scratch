import pickle
from multiprocessing import Pool

import numpy as np

from bpe import Tokenizer, train_bpe

EOT = "<|endoftext|>"


def load_tokenizer():
    global tok
    tok = Tokenizer(**pickle.load(open("data/tokenizer.pkl", "rb")))


def encode(text):
    return np.array(tok.encode(text), dtype=np.uint16)


def encode_file(path, out):
    stories = open(path).read().split(EOT)
    # 10k stories per chunk, encoded in parallel
    chunks = [EOT.join(stories[i:i + 10_000]) + EOT for i in range(0, len(stories), 10_000)]
    with Pool(initializer=load_tokenizer) as pool:
        np.save(out, np.concatenate(pool.map(encode, chunks)))


if __name__ == "__main__":
    # BPE is trained on the first 100MB only; more barely changes the merges
    text = open("data/TinyStoriesV2-GPT4-train.txt").read(100_000_000)
    vocab, merges = train_bpe(text, 10_000, [EOT])
    pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": [EOT]}, open("data/tokenizer.pkl", "wb"))

    encode_file("data/TinyStoriesV2-GPT4-valid.txt", "data/valid.npy")
    encode_file("data/TinyStoriesV2-GPT4-train.txt", "data/train.npy")
