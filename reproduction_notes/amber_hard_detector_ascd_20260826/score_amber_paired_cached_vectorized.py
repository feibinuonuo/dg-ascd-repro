#!/usr/bin/env python3
"""Cached-document launcher for the frozen vectorized AMBER scorer.

It memoizes only repeated spaCy document construction.  All AMBER rules and
the scorer's arithmetic remain the imported implementation.
"""

from __future__ import annotations

import spacy

import score_amber_paired_vectorized as scorer


class CachedNLP:
    def __init__(self, nlp):
        self.nlp = nlp
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = self.nlp(text)
        return self.cache[text]


def main() -> None:
    original_load = spacy.load
    scorer.spacy.load = lambda name: CachedNLP(original_load(name))
    scorer.main()


if __name__ == "__main__":
    main()
