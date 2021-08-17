#!/usr/bin/env python3

import nltk
import unicodedata

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.probability import FreqDist


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english', minimum=2, maximum=200):
        self.min = minimum
        self.max = maximum
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if token in list(self.reduced)
               and not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        words = []
        docs = []
        for document in documents:
            docs.append(document)
            for para in document:
                for sent in para:
                    for token, tag in sent:
                        words.append(token)

        counts = FreqDist(words)
        self.reduced = set(
            w for w in words if counts[w] > self.min and counts[w] < self.max
        )

        return [
            ' '.join(self.normalize(doc)) for doc in docs
        ]


if __name__ == '__main__':
    from reader import HTMLPickledCorpusReader
    corpus = HTMLPickledCorpusReader('../mini_food_corpus_proc')

    normalizer = TextNormalizer(minimum=10, maximum=50)
    print(normalizer.fit_transform(corpus.docs()))