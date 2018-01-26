# mp_train
# Parallel fit of models
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat Dec 16 08:04:57 2017 -0500
#
# ID: mp_train.py [] benjamin@bengfort.com $

"""
Parallel fit of models
"""

##########################################################################
## Imports
##########################################################################

import time
import logging
import multiprocessing as mp

from functools import wraps

from reader import PickledCorpusReader
from transformers import TextNormalizer, identity

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(processName)-10s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


def documents(corpus):
    return [
        list(corpus.docs(fileids=fileid))
        for fileid in corpus.fileids()
    ]


def labels(corpus):
    return [
        corpus.categories(fileids=fileid)[0]
        for fileid in corpus.fileids()
    ]


@timeit
def train_model(path, model, saveto=None, cv=12):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the cv parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """
    # Load the corpus data and labels for classification
    corpus = PickledCorpusReader(path)
    X = documents(corpus)
    y = labels(corpus)

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores


def fit_naive_bayes(path, saveto=None, cv=12):

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', MultinomialNB())
    ])

    if saveto is None:
        saveto = "naive_bayes_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "naive bayes training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


def fit_logistic_regression(path, saveto=None, cv=12):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', LogisticRegression())
    ])

    if saveto is None:
        saveto = "logistic_regression_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "logistic regression training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


def fit_multilayer_perceptron(path, saveto=None, cv=12):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', MLPClassifier(hidden_layer_sizes=(10,10), early_stopping=True))
    ])

    if saveto is None:
        saveto = "multilayer_perceptron_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "multilayer perceptron training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


@timeit
def sequential(path):
    #Run each fit one after the other
    fit_naive_bayes(path)
    fit_logistic_regression(path)
    fit_multilayer_perceptron(path)


@timeit
def parallel(path):
    tasks = [
        fit_naive_bayes, fit_logistic_regression, fit_multilayer_perceptron,
    ]

    procs = []
    for task in tasks:
        proc = mp.Process(name=task.__name__, target=task, args=(path,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    path = "../corpus"
    # print("beginning sequential tasks")
    # _, delta = sequential(path)
    # print("total sequential fit time: {:0.2f} seconds".format(delta))

    logger.info("beginning parallel tasks")
    _, delta = parallel(path)
    logger.info("total parallel fit time: {:0.2f} seconds".format(delta))
