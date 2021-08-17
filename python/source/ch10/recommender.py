#!/usr/bin/env python3

import os
import time
import argparse

from functools import wraps
from nltk import wordpunct_tokenize

from sklearn.externals import joblib
from sklearn.neighbors import BallTree, KDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors, LSHForest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

from transformer import TextNormalizer
from reader import HTMLPickledCorpusReader


class KNNTransformer(NearestNeighbors, TransformerMixin):
    """
    Scikit-Learn's KNN doesn't have a transform method,
    so give it one.
    """
    def __init__(self, k=3, **kwargs):
        """
        Note: tried LSHForest, still too slow
        :param k:
        :param kwargs:
        """
        self.model = NearestNeighbors(n_neighbors=k, **kwargs)

    def fit(self, documents):
        self.model.fit(documents)
        return self

    def transform(self, documents):
        return [
            self.model.kneighbors(document)
            for document in documents
        ]


class BallTreeTransformer(NearestNeighbors, TransformerMixin):
    """
    Scikit-Learn's BallTree doesn't have a transform method,
    so give it one.

    Note: didn't end up needing this
    """
    def __init__(self, **kwargs):
        self.model = None

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            BallTree(documents)
        ]

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


class KNNRecommender(object):
    """
    Given input terms, provide k recipe recommendations

    Note: didn't end up using this one because it was too slow.
    """
    def __init__(self, k=3, **kwargs):
        self.k = k
        self.pipeline = Pipeline([
            ('norm', TextNormalizer(minimum=10, maximum=100)),
            ('tfidf', TfidfVectorizer()),
            ('knn', Pipeline([
                ('svd', TruncatedSVD(n_components=100)),
                ('model', KNNTransformer(k=self.k, algorithm='ball_tree'))
            ]))
        ])

        self.lex_path = "lexicon.pkl"
        self.vect_path = "vect.pkl"
        self.vectorizer = False
        self.lexicon = None
        self.load()

    def load(self):
        """
        Load a pickled vectorizer and vectorized corpus from disk,
        if they exist.
        """
        if os.path.exists(self.vect_path):
            joblib.load(open(self.vect_path, 'rb'))
            joblib.load(open(self.lex_path, 'rb'))
        else:
            self.vectorizer = False
            self.lexicon = None

    def save(self):
        """
        It takes a long time to fit, so just do it once!
        """
        joblib.dump(self.vect, open(self.vect_path, 'wb'))
        joblib.dump(self.lexicon, open(self.lex_path, 'wb'))

    def fit_transform(self, documents):
        # Vectorizer will be False if pipeline hasn't been fit yet,
        # Trigger fit_transform and save the vectorizer and lexicon.
        if self.vectorizer == False:
            self.lexicon = self.pipeline.fit_transform(documents)
            self.vect = self.pipeline.named_steps['tfidf']
            self.knn = self.pipeline.named_steps['knn']
            self.save()
        # If there's a stored vectorizer and prefitted lexicon,
        # use them instead.
        else:
            self.vect = self.vectorizer
            self.knn = Pipeline([
                ('svd', TruncatedSVD(n_components=100)),
                ('knn', KNNTransformer(k=self.k, algorithm='ball_tree'))
            ])
            self.knn.fit_transform(self.lexicon)

    def recommend(self, terms):
        """
        Given input list of ingredient terms,
        return the k closest matching recipes.

        :param terms: list of strings
        :return: list of document indices of documents
        """
        vect_doc = self.vect.transform(wordpunct_tokenize(terms))
        distance_matches = self.knn.transform(vect_doc)
        # the result is a list with a 2-tuple of arrays
        matches = distance_matches[0][1][0]
        # the matches are the indices of documents
        return matches


class BallTreeRecommender(object):
    """
    Given input terms, provide k recipe recommendations
    """
    def __init__(self, k=3, **kwargs):
        self.k = k
        self.trans_path = "svd.pkl"
        self.tree_path = "tree.pkl"
        self.transformer = False
        self.tree = None
        self.load()

    def load(self):
        """
        Load a pickled transformer and tree from disk,
        if they exist.
        """
        if os.path.exists(self.trans_path):
            self.transformer = joblib.load(open(self.trans_path, 'rb'))
            self.tree = joblib.load(open(self.tree_path, 'rb'))
        else:
            self.transformer = False
            self.tree = None

    def save(self):
        """
        It takes a long time to fit, so just do it once!
        """
        joblib.dump(self.transformer, open(self.trans_path, 'wb'))
        joblib.dump(self.tree, open(self.tree_path, 'wb'))

    def fit_transform(self, documents):
        # Transformer will be False if pipeline hasn't been fit yet,
        # Trigger fit_transform and save the transformer and lexicon.
        if self.transformer == False:
            self.transformer = Pipeline([
                ('norm', TextNormalizer(minimum=50, maximum=200)),
                ('transform', Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('svd', TruncatedSVD(n_components=200))
                ])
                 )
            ])
            self.lexicon = self.transformer.fit_transform(documents)
            self.tree = BallTree(self.lexicon)
            self.save()

    def query(self, terms):
        """
        Given input list of ingredient terms,
        return the k closest matching recipes.

        :param terms: list of strings
        :return: list of document indices of documents
        """
        vect_doc = self.transformer.named_steps['transform'].fit_transform(
            wordpunct_tokenize(terms)
        )
        dists, inds = self.tree.query(vect_doc, k=self.k)
        return inds[0]

@timeit
def suggest_recipe(query):
    """
    Quick wrapper for KNNRecommender.recommend()
    :param query:
    :return:
    """
    corpus = HTMLPickledCorpusReader('../mini_food_corpus_proc')
    docs = list(corpus.docs())
    titles = list(corpus.titles())
    tree = BallTreeRecommender(k=3)
    tree.fit_transform(docs)
    results = tree.query(query)
    return [titles[result] for result in results]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ingredients', help='ingredients to parse, surround by quotes')
    args = parser.parse_args()
    recs, build_time = suggest_recipe(args.ingredients)
    print("Here are some recipes related to {}:".format(
        args.ingredients)
    )
    for rec in recs:
        print(rec)

    print("Build time: {}".format(build_time))

    # start = time.time()
    # print("loading corpus...")
    # corpus = HTMLPickledCorpusReader('../mini_food_corpus_proc')
    # titles = list(corpus.titles())
    # print("corpus load time:{}".format(time.time() - start))
    #
    #
    #
    # # inter = time.time()
    # # print("prepping docs...")
    # # docs = list(corpus.docs())
    # # print("doc prep time:{}".format(time.time() - inter))
    # # inter = time.time()
    # # print("normalizing docs...")
    # # normed_docs = TextNormalizer().fit_transform(docs)
    # # print("text norm fit time:{}".format(time.time() - inter))
    # # inter = time.time()
    # # print("vectorizing docs...")
    # # vect_docs = TfidfVectorizer().fit_transform(normed_docs)
    # # print("tfidf fit time:{}".format(time.time() - inter))
    # # inter = time.time()
    # # print("truncating docs...")
    # # trunc_docs = TruncatedSVD().fit_transform(vect_docs)
    # # print("svd fit time:{}".format(time.time() - inter))
    # # print("fitting ball tree...")
    # # tree = BallTree(trunc_docs)
    # # print("pickling tree...")
    # # joblib.dump(tree, open('tree.pkl', 'wb'))
    # # print("pickling transformed corpus...")
    # # joblib.dump(trunc_docs, open('svd_lexicon.pkl', 'wb'))
    # # print("ball tree fit time:{}".format(time.time() - start))
    #

    # terms = "tortillas cilantro chicken thighs"
    #
    # # tree = joblib.load(open('tree.pkl', 'rb'))
    # transformer = joblib.load(open('svd.pkl', 'rb'))
    # print(transformer.transform(wordpunct_tokenize(terms)))
    # dists, inds = tree.query(transformer.transform(wordpunct_tokenize(terms)), k=3)
    # for ind in inds[0]:
    #     print(titles[ind])
    # print("final build time:{}".format(time.time() - start))