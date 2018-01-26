import os
import nltk
import unicodedata
import numpy as np

from itertools import groupby
from operator import itemgetter

from reader import PickledCorpusReader

from nltk.corpus import wordnet as wn
from nltk.cluster import KMeansClusterer

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()


def is_punct(token):
    # Is every character punctuation?
    return all(
        unicodedata.category(char).startswith('P')
        for char in token
    )


def wnpos(tag):
    # Return the WordNet POS tag from the Penn Treebank tag
    return {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)


def normalize(document, stopwords=STOPWORDS):
    """
    Removes stopwords and punctuation, lowercases, lemmatizes
    """

    for token, tag in document:
        token = token.lower().strip()

        if is_punct(token) or (token in stopwords):
            continue

        yield lemmatizer.lemmatize(token, wnpos(tag))


class KMeansTopics(object):

    def __init__(self, corpus, k=10):
        """
        corpus is a corpus object, e.g. an HTMLCorpusReader()
        or an HTMLPickledCorpusReader() object

        k is the number of clusters
        """
        self.k = k
        self.model = None
        self.vocab = list(
            set(normalize(corpus.words(categories=['news'])))
            )

    def vectorize(self, document):
        """
        Vectorizes a document consisting of a list of part of speech
        tagged tokens using the segmentation and tokenization methods.

        One-hot encode the set of documents
        """
        features = set(normalize(document))
        return np.array([
            token in features for token in self.vocab], np.short)

    def cluster(self, corpus):
        """
        Fits the K-Means model to the given data.
        """
        cosine = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(
            self.k, distance=cosine, avoid_empty_clusters=True)
        self.model.cluster([
            self.vectorize(
                corpus.words(fileid)
            ) for fileid in corpus.fileids(categories=['news'])
        ])

    def classify(self, document):
        """
        Pass through to the internal model classify
        """
        return self.model.classify(self.vectorize(document))

if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')

    clusterer = KMeansTopics(corpus, k=7)
    clusterer.cluster(corpus)

    # Classify documents in the new corpus by cluster affinity
    groups  = [
        (clusterer.classify(corpus.words(fileid)), fileid)
        for fileid in corpus.fileids(categories=['news'])
    ]

    # Group documents in corpus by cluster and display them
    groups.sort(key=itemgetter(0))
    for group, items in groupby(groups, key=itemgetter(0)):
        for cluster, fname in items:
            print("Cluster {}: {}".format(cluster+1,fname))
