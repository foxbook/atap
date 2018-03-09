#!/usr/bin/env python3

import nltk

from math import log
from collections import Counter, defaultdict

from nltk.util import ngrams
from nltk.probability import ProbDistI, FreqDist, ConditionalFreqDist

from reader import PickledCorpusReader


def count_ngrams(n, vocabulary, texts):
    counter = NgramCounter(n, vocabulary)
    counter.train_counts(texts)
    return counter


class NgramCounter(object):
    """
    The NgramCounter class counts ngrams given a vocabulary and ngram size.
    """

    def __init__(self, n, vocabulary, unknown="<UNK>"):
        """
        n is the size of the ngram
        """
        if n < 1:
            raise ValueError("ngram size must be greater than or equal to 1")

        self.n = n
        self.unknown = unknown
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }

        self.vocabulary = vocabulary
        self.allgrams = defaultdict(ConditionalFreqDist)
        self.ngrams = FreqDist()
        self.unigrams = FreqDist()

    def train_counts(self, training_text):
        for sent in training_text:
            checked_sent = (self.check_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                self.ngrams[ngram] += 1
                context, word = tuple(ngram[:-1]), ngram[-1]
                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False

                for window, ngram_order in enumerate(range(self.n, 1, -1)):
                    context = context[window:]
                    self.allgrams[ngram_order][context][word] += 1
                self.unigrams[word] += 1

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unknown

    def to_ngrams(self, sequence):
        """
        Wrapper for NLTK ngrams method
        """
        return ngrams(sequence, self.n, **self.padding)


class BaseNgramModel(object):
    """
    The BaseNgramModel creates an n-gram language model.
    This base model is equivalent to a Maximum Likelihood Estimation.
    """

    def __init__(self, ngram_counter):
        """
        BaseNgramModel is initialized with an NgramCounter.
        """
        self.n = ngram_counter.n
        self.ngram_counter = ngram_counter
        self.ngrams = ngram_counter.ngrams
        self._check_against_vocab = self.ngram_counter.check_against_vocab

    def check_context(self, context):
        """
        Ensures that the context is not longer than or equal to the model's
        n-gram order.

        Returns the context as a tuple.
        """
        if len(context) >= self.n:
            raise ValueError("Context too long for this n-gram")

        return tuple(context)

    def score(self, word, context):
        """
        For a given string representation of a word, and a string word context,
        returns the maximum likelihood score that the word will follow the
        context.
        """
        context = self.check_context(context)

        return self.ngrams[context].freq(word)

    def logscore(self, word, context):
        """
        For a given string representation of a word, and a word context,
        computes the log probability of this word in this context.
        """
        score = self.score(word, context)
        if score == 0.0:
            return float("-inf")

        return log(score, 2)

    def entropy(self, text):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given text represented as a list of comma-separated strings.
        This is the average log probability of each word in the text.
        """
        normed_text = (self._check_against_vocab(word) for word in text)
        entropy = 0.0
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            entropy += self.logscore(word, context)
            processed_ngrams += 1
        return - (entropy / processed_ngrams)

    def perplexity(self, text):
        """
        Given list of comma-separated strings, calculates the perplexity
        of the text.
        """
        return pow(2.0, self.entropy(text))


class AddKNgramModel(BaseNgramModel):
    """
    Provides Add-k-smoothed scores.
    """

    def __init__(self, k, *args):
        """
        Expects an input value, k, a number by which
        to increment word counts during scoring.
        """
        super(AddKNgramModel, self).__init__(*args)

        self.k = k
        self.k_norm = len(self.ngram_counter.vocabulary) * k

    def score(self, word, context):
        """
        With Add-k-smoothing, the score is normalized with
        a k value.
        """
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        context_count = context_freqdist.N()
        return (word_count + self.k) / \
               (context_count + self.k_norm)


class LaplaceNgramModel(AddKNgramModel):
    """
    Implements Laplace (add one) smoothing.
    Laplace smoothing is the base case of Add-k smoothing,
    with k set to 1.
    """
    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)


class KneserNeyModel(BaseNgramModel):
    """
    Implements Kneser-Ney smoothing
    """
    def __init__(self, *args):
        super(KneserNeyModel, self).__init__(*args)
        self.model = nltk.KneserNeyProbDist(self.ngrams)

    def score(self, word, context):
        """
        Use KneserNeyProbDist from NLTK to get score
        """
        trigram = tuple((context[0], context[1], word))
        return self.model.prob(trigram)

    def samples(self):
        return self.model.samples()

    def prob(self, sample):
        return self.model.prob(sample)


if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')
    tokens = [''.join(word) for word in corpus.words()]
    vocab = Counter(tokens)
    sents = list([word[0] for word in sent] for sent in corpus.sents())

    counter = count_ngrams(3, vocab, sents)
    knm = KneserNeyModel(counter)


    def complete(input_text):
        tokenized = nltk.word_tokenize(input_text)
        if len(tokenized) < 2:
            response = "Say more."
        else:
            completions = {}
            for sample in knm.samples():
                if (sample[0], sample[1]) == (tokenized[-2], tokenized[-1]):
                    completions[sample[2]] = knm.prob(sample)
            if len(completions) == 0:
                response = "Can we talk about something else?"
            else:
                best = max(
                    completions.keys(), key=(lambda key: completions[key])
                )
                tokenized += [best]
                response = " ".join(tokenized)

        return response

    print(complete("The President of the United"))
    print(complete("This election year will"))
