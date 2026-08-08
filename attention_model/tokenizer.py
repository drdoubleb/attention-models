"""A miniature byte-pair-encoding (BPE) tokenizer, written for clarity.

Real LLM tokenizers (GPT-2's BPE, tiktoken, SentencePiece) work on raw
bytes and pre-split text into words, but the core algorithm is exactly
what is implemented here:

  1. Start with the text split into single characters.
  2. Count every adjacent pair of tokens.
  3. Merge the most frequent pair into a new, longer token.
  4. Repeat for a fixed number of merges.

Encoding new text replays the learned merges in the same order.

Everything the animations need is recorded along the way:
  * ``merge_history``  - one entry per learned merge (pair, new token, count)
  * ``encode_with_history`` - the intermediate token lists while encoding a
    given text, so the video can show tokens fusing step by step.
"""

from collections import Counter


class MiniBPE:
    def __init__(self):
        self.merges = []          # list of (left, right) pairs, in learned order
        self.vocab = []           # every token string, index = token id
        self.token_to_id = {}
        self.merge_history = []   # [{pair, new_token, count}] for the animation

    # ------------------------------------------------------------ training
    def train(self, text, num_merges):
        """Learn ``num_merges`` merges from ``text``."""
        tokens = list(text)  # step 1: one token per character

        # The starting vocabulary is every distinct character, sorted so
        # token ids are stable across runs.
        self.vocab = sorted(set(tokens))
        self.merges = []
        self.merge_history = []

        for _ in range(num_merges):
            pair_counts = Counter(zip(tokens, tokens[1:]))
            if not pair_counts:
                break
            (left, right), count = pair_counts.most_common(1)[0]
            if count < 2:
                break  # nothing repeats any more; further merges are pointless

            new_token = left + right
            tokens = self._merge_once(tokens, left, right)
            self.merges.append((left, right))
            self.vocab.append(new_token)
            self.merge_history.append(
                {"pair": (left, right), "new_token": new_token, "count": count}
            )

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        return self

    @staticmethod
    def _merge_once(tokens, left, right):
        """Replace every adjacent (left, right) with the fused token."""
        out = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                out.append(left + right)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ------------------------------------------------------------ encoding
    def encode_tokens(self, text):
        """Encode ``text`` and return the token *strings*."""
        tokens = list(text)
        for left, right in self.merges:
            tokens = self._merge_once(tokens, left, right)
        return tokens

    def encode(self, text):
        """Encode ``text`` and return the token *ids*."""
        return [self.token_to_id[t] for t in self.encode_tokens(text)]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def encode_with_history(self, text):
        """Encode ``text`` while recording every intermediate state.

        Returns a list of steps.  The first step is the raw character
        split; each later step is the token list after one learned merge
        actually changed something, plus which pair was merged.
        """
        tokens = list(text)
        steps = [{"tokens": list(tokens), "merged_pair": None}]
        for left, right in self.merges:
            new_tokens = self._merge_once(tokens, left, right)
            if new_tokens != tokens:
                tokens = new_tokens
                steps.append({"tokens": list(tokens), "merged_pair": (left, right)})
        return steps

    @property
    def vocab_size(self):
        return len(self.vocab)
