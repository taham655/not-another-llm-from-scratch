from collections import Counter, defaultdict

import regex

# GPT-2 pre-tokenization: merges never cross word boundaries
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_on_specials(text, special_tokens):
    if not special_tokens:
        return [text]
    # longest first, so overlapping specials match the longer one
    alt = "|".join(regex.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    return [c for c in regex.split(f"({alt})", text) if c]


def train_bpe(text, vocab_size, special_tokens=()):
    vocab = {i: bytes([i]) for i in range(256)}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode()

    word_counts = Counter()
    for chunk in split_on_specials(text, list(special_tokens)):
        if chunk in special_tokens:
            continue
        for w in regex.findall(PAT, chunk):
            word_counts[tuple(bytes([b]) for b in w.encode())] += 1

    words = list(word_counts)
    freqs = list(word_counts.values())

    # which words contain each pair, so a merge only touches words that change
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freqs[wi]
            pair_to_words[pair].add(wi)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        # most frequent pair, ties broken by the lexicographically greatest
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        new_tok = best[0] + best[1]
        vocab[len(vocab)] = new_tok

        for wi in list(pair_to_words[best]):
            old, f = words[wi], freqs[wi]
            new, i = [], 0
            while i < len(old):
                if i + 1 < len(old) and (old[i], old[i + 1]) == best:
                    new.append(new_tok)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            new = tuple(new)
            words[wi] = new
            for p in zip(old, old[1:]):
                pair_counts[p] -= f
                pair_to_words[p].discard(wi)
            for p in zip(new, new[1:]):
                pair_counts[p] += f
                pair_to_words[p].add(wi)
        pair_counts = +pair_counts  # drop zero counts

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=()):
        self.vocab = vocab
        self.id_of = {b: i for i, b in vocab.items()}
        self.rank = {pair: r for r, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens)
        self.cache = {}  # words repeat constantly, so encode each one once
        for s in self.special_tokens:
            if s.encode() not in self.id_of:
                self.vocab[len(self.vocab)] = s.encode()
                self.id_of[s.encode()] = len(self.vocab) - 1

    def _encode_word(self, word):
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            # replay merges in the order they were learned
            pair = min(zip(parts, parts[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(pair[0] + pair[1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [self.id_of[p] for p in parts]

    def encode(self, text):
        ids = []
        for chunk in split_on_specials(text, self.special_tokens):
            if chunk in self.special_tokens:
                ids.append(self.id_of[chunk.encode()])
            else:
                for w in regex.findall(PAT, chunk):
                    if w not in self.cache:
                        self.cache[w] = self._encode_word(w.encode())
                    ids.extend(self.cache[w])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    text = "the cat in the hat sat on the mat. the cat ate the rat!"
    vocab, merges = train_bpe(text, vocab_size=256 + 5)
    print("merges:", merges)
    assert merges[0] == (b"a", b"t")
    tok = Tokenizer(vocab, merges)
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) < len(text.encode())

    s = "héllo 世界 🌍 the theatre"
    assert tok.decode(tok.encode(s)) == s

    eot = "<|endoftext|>"
    vocab, merges = train_bpe(f"the cat{eot}the cat", 256 + 3, [eot])
    tok = Tokenizer(vocab, merges, [eot])
    assert tok.encode(f"a{eot}b") == [tok.id_of[b"a"], tok.id_of[eot.encode()], tok.id_of[b"b"]]

    _, m = train_bpe("ab cd", 257)
    assert m[0] == (b"c", b"d")

    print("all checks passed")
