#!/usr/bin/env python3

import re
import nltk

from reader import PickledCorpusReader

GRAMMAR = """
    S -> NNP VP
    VP -> V PP
    PP -> P NP
    NP -> DT N
    NNP -> 'Gwen' | 'George'
    V -> 'looks' | 'burns'
    P -> 'in' | 'for'
    DT -> 'the'
    N -> 'castle' | 'ocean'
    """

cfg = nltk.CFG.fromstring(GRAMMAR)

print(cfg)
print(cfg.start())
print(cfg.productions())
