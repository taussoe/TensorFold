"""Drafts copied from the context: when the last tokens also stand earlier in the prompt or the reply, the tokens that
followed them there are proposed (a model rewriting a file, quoting code or a tool result repeats long runs).

Drafts only propose: every round verifies them against the model's own keyed samples, so replies stay the ones
serial decoding gives.
"""

from __future__ import annotations

import os

import numpy as np

GRAM = 3        # tokens a match is found by
# Measured on file rewrites (GLM-5.3-Flash, two Sparks): 7 drafts once 8 tokens agree gave 1.12-1.38x the MTP-only
# speed; 5 drafts 1.05-1.18x; 5 agreeing tokens also copied into fresh code (3% slower there)
AGREE = int(os.environ.get("TF_GLM_LOOKUP_AGREE", "8"))    # tokens that must agree (the gram and before)
DRAFTS = int(os.environ.get("TF_GLM_LOOKUP_DRAFTS", "7"))  # at most (the widest verify window, 8 rows)
REACH = 16      # tokens compared backwards
SHIFT = 21      # a 3-gram's key: its ids packed into one int (ids below 2**20; keyed image ids are negative)


def _key(a: int, b: int, c: int) -> int:
    return (a << (2 * SHIFT)) + (b << SHIFT) + c


class PromptLookup:
    """An index of the context's 3-grams (their latest start); ``propose`` after ``extend``. Keys are plain ints: a
    dict of tuples made 18k tracked objects a request, and the garbage collector once took 1.4 s over them."""

    def __init__(self, prompt, *, agree: int = AGREE, drafts: int = DRAFTS) -> None:
        self.ids: list[int] = [int(t) for t in prompt]
        self.agree, self.most = agree, drafts
        self.index: dict[int, int] = {}
        a = np.asarray(self.ids, dtype=np.int64)
        if len(a) > GRAM:
            # every gram whose next token is known, latest start last (so it wins)
            keys = (a[:-GRAM] << (2 * SHIFT)) + (a[1:-GRAM + 1] << SHIFT) + a[2:-GRAM + 2]
            self.index = dict(zip(keys.tolist(), range(len(keys))))

    def extend(self, tokens) -> None:
        for t in tokens:
            self.ids.append(int(t))
            j = len(self.ids) - 1                  # the gram ending just before t now has a next token: t
            if j >= GRAM:
                self.index[_key(*self.ids[j - GRAM:j])] = j - GRAM

    def propose(self, room: int) -> list[int]:
        """Up to ``room`` (and DRAFTS) tokens that followed the last tokens earlier, when at least AGREE agree."""

        ids = self.ids
        if len(ids) < max(GRAM, self.agree) or room <= 0:
            return []
        start = self.index.get(_key(*ids[-GRAM:]))
        if start is None:
            return []
        end = start + GRAM                          # the earlier run's next token
        n = len(ids)
        k = GRAM
        while k < REACH and start - (k - GRAM) - 1 >= 0 and ids[start - (k - GRAM) - 1] == ids[n - k - 1]:
            k += 1
        if k < self.agree:
            return []
        stop = min(end + min(room, self.most), n - 1)
        return ids[end:stop] if stop > end else []
