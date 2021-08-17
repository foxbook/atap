#!/usr/bin/env python3

import time
import numpy as np
from functools import wraps

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper

def documents(corpus):
    return list(corpus.reviews())

def continuous(corpus):
    return list(corpus.scores())

def make_categorical(corpus):
    """
    terrible : 0.0 < y <= 3.0
    okay     : 3.0 < y <= 5.0
    great    : 5.0 < y <= 7.0
    amazing  : 7.0 < y <= 10.1
    :param corpus:
    :return:
    """
    return np.digitize(continuous(corpus), [0.0, 3.0, 5.0, 7.0, 10.1])

@timeit
def train_model(path, model, continuous=True, saveto=None, cv=12):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the cv parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """
    # Load the corpus data and labels for classification
    corpus = PickledReviewsReader(path)
    X = documents(corpus)
    if continuous:
        y = continuous(corpus)
        scoring = 'r2_score'
    else:
        y = make_categorical(corpus)
        scoring = 'f1_score'

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    from reader import PickledReviewsReader
    from transformer import TextNormalizer, KeyphraseExtractor

    cpath = '../review_corpus_proc'
    mpath = 'ann_cls.pkl'

    pipeline = Pipeline([
        ('norm', TextNormalizer()), # can use KeyphraseExtractor() instead
        ('tfidf', TfidfVectorizer()),
        ('ann', MLPClassifier(hidden_layer_sizes=[500,150], verbose=True))
    ])

    print("Starting training...")
    scores, delta = train_model(cpath, pipeline, continuous=False, saveto=mpath)

    print("Training complete.")
    for idx, score in enumerate(scores):
        print("Accuracy on slice #{}: {}.".format((idx+1), score))
    print("Total fit time: {:0.2f} seconds".format(delta))
    print("Model saved to {}.".format(mpath))
