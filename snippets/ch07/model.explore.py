#!/usr/bin/env python3

import math

from collections import defaultdict

from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist

LPAD_SYMBOL = "<s>"
RPAD_SYMBOL = "</s>"


class NgramCounter(object):

    def __init__(self, N, lpad=LPAD_SYMBOL, rpad=RPAD_SYMBOL):

        self.n = N
        self.padding = {
            "pad_left": bool(lpad),
            "left_pad_symbol": lpad,
            "pad_right": bool(rpad),
            "right_pad_symbol": rpad,
        }

        # Internal frequency data structures
        self.unigrams = FreqDist()
        self.allgrams = defaultdict(ConditionalFreqDist)

    def to_ngrams(self, sentence):
        """
        Computes the N-grams of a sentence given the supplied N and padding.
        """
        return ngrams(sentence, self.n, **self.padding)

    def fit(self, sentences):
        """
        Count the ngrams and relative contexts of words.
        """
        for sent in sentences:
            # Compute unigram frequencies for each word in the sentence
            for word in sent:
                self.unigrams[word] += 1

            # Compute conditional frequencies for all contexts up to order N
            for ngram in self.to_ngrams(sent):

                # The context is everything up to the last word
                context, word = tuple(ngram[:-1]), ngram[-1]

                # Compute every n-gram size from N down to 1 and count toward
                # the conditional frequency.
                for index, ngram_order in enumerate(range(self.n, 1, -1)):
                    subcontext = context[index:]
                    self.allgrams[ngram_order][subcontext][word] += 1


class BaseNGramModel(NgramCounter):

    def score(self, word, context):
        return self.allgrams[self.n][context].freq(word)

    def logscore(self, word, context):
        score = self.score(word, context)
        if score == 0.0:
            return float("-inf")
        return math.log(score, 2)

    def entropy(self, text):
        entropy = 0.0
        count = 0

        for ngram in self.to_ngrams(text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            entropy += self.logscore(word, context)
            count += 1

        return - (entropy / count)

    def perplexity(self, text):
        return math.pow(2.0, self.entropy(text))


if __name__ == '__main__':
    # from reader import PickledCorpusReader
    #
    # corpus = PickledCorpusReader('../corpus')
    model = BaseNGramModel(3)

    from nltk import word_tokenize
    sents = map(word_tokenize, [
        "The red fox jumped over the brown dog.",
        "See Spot run, run, Spot, run!",
        "A little dog laughed to see such sport, and the fork ran away with the spoon.",
    ])

    model.fit(sents)

    test = list(word_tokenize("A little dog jumped over the red fox."))
