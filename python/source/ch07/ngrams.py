#!/usr/bin/env python3

import sys
import argparse

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import ngrams as nltk_ngrams
from functools import partial

LPAD_SYMBOL = "<s>"
RPAD_SYMBOL = "</s>"

nltk_ngrams = partial(nltk_ngrams,
    pad_right=True, pad_left=True,
    right_pad_symbol=RPAD_SYMBOL, left_pad_symbol=LPAD_SYMBOL
)


def ngrams(words, n=2):
    for idx in range(len(words)-n+1):
        yield tuple(words[idx:idx+n])


def ngrams2(text, n=2):
    for sent in sent_tokenize(text):
        sent = word_tokenize(sent)
        for ngram in nltk_ngrams(sent, n):
            yield ngram


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nltk", action="store_true", help="use nltk method")
    parser.add_argument("-n", type=int, default=2, help="ngram size to compute")
    parser.add_argument("phrase", help="surround a single phrase in quotes")
    args = parser.parse_args()

    if args.nltk:
        for gram in ngrams2(args.phrase, args.n):
            print(gram)
    else:
        for gram in ngrams(word_tokenize(args.phrase), args.n):
            print(gram)
