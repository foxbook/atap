#!/usr/bin/env python3
"""
Performs entity extraction using NLTK
"""
import os
import time
import json

from nltk import ne_chunk
from nltk.chunk import tree2conlltags

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])
# GPE is Geo-Political Entity, GSP is Geo-Socio-Political group


def identity(words):
    return words


class EntityExtractor(BaseEstimator, TransformerMixin):
    """
    Perform entity extraction

    Output is saved
    """
    def __init__(self, labels=GOODLABELS, **kwargs):
        self.labels = labels

    def get_entities(self, document):
        """
        Extract entities from a single document using the
        nltk.tree.ne_chunk method

        This method is called multiple times by the tranform method

        :param document: a list of lists of tuples
        :return entities: a list of comma-separated strings
        """
        entities = []
        for paragraph in document:
            for sentence in paragraph:
                # classifier chunk the sentences, adds category labels, e.g. PERSON
                trees = ne_chunk(sentence)
                # select only trees with the kinds of entities we want
                for tree in trees:
                    if hasattr(tree, 'label'):
                        if tree.label() in self.labels:
                            # entities is a list, each entry is a list of entities
                            # for a document
                            entities.append(
                                ' '.join([child[0].lower() for child in tree])
                                )
        return entities

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """
        Create a representation of the documents as a list of their entities
        """
        for document in documents:
            yield self.get_entities(document[0])

def create_pipeline(estimator):
    steps = [
        ('extract', EntityExtractor()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False
        )),
        ('classifier', estimator)
    ]

    return Pipeline(steps)

def score_models(models, loader):
    for model in models:

        name = model.named_steps['classifier'].__class__.__name__
        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }

        for X_train, X_test, y_train, y_test in loader:
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        yield scores

if __name__ == '__main__':
    from loader import CorpusLoader
    from reader import PickledCorpusReader

    reader = PickledCorpusReader('../corpus')
    # extractor = EntityExtractor()
    # print(list(extractor.fit_transform(reader.docs())))

    labels = ['books', 'cinema', 'cooking', 'gaming', 'sports', 'tech']
    loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

    forms = (LogisticRegression, SGDClassifier, MultinomialNB, GaussianNB)
    models = []
    for form in forms:
        models.append(create_pipeline(form()))

    for scores in score_models(models, loader):
        with open('results_with_ner.json', 'a') as f:
            f.write(json.dumps(scores) + "\n")
