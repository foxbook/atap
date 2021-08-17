import os
import time
import numpy as np

from functools import wraps

from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score

from keras.layers.embeddings import Embedding
from keras.models import load_model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation, LSTM


N_FEATURES = 10000
DOC_LEN = 60
N_CLASSES = 2

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
    return np.digitize(continuous(corpus), [0.0, 3.0, 5.0, 7.0, 10.1])

def binarize(corpus):
    return np.digitize(continuous(corpus), [0.0, 3.0, 5.1])

def build_nn():
    """
    Create a function that returns a compiled neural network
    :return: compiled Keras neural network model
    """
    nn = Sequential()
    nn.add(Dense(500, activation='relu', input_shape=(N_FEATURES,)))
    nn.add(Dense(150, activation='relu'))
    nn.add(Dense(N_CLASSES, activation='softmax'))
    nn.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return nn

def build_lstm():
    lstm = Sequential()
    lstm.add(Embedding(N_FEATURES+1, 128, input_length=DOC_LEN))
    lstm.add(Dropout(0.4))
    lstm.add(LSTM(units=200, recurrent_dropout=0.2, dropout=0.2))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(N_CLASSES, activation='sigmoid'))
    lstm.compile(
        loss='categorical_crossentropy', # b/c target vals are 1 or 2
        optimizer='adam',
        metrics=['accuracy']
    )
    return lstm

@timeit
def train_model(path, model, saveto=None, cv=12, **kwargs):
    """
    Trains model from corpus at specified path;
    fitting the model on the full data and
    writing it to disk at the saveto directory if specified.
    Returns the scores.
    """
    # Load the corpus data and labels for classification
    # corpus = PickledReviewsReader(path) # for Pitchfork
    corpus = PickledAmazonReviewsReader(path)
    X = documents(corpus)
    # y = categorical(corpus) # for Pitchfork
    y = binarize(corpus)

    # Compute cross validation scores
    # mp note: http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        # have to save the keras part using keras' save method
        model.steps[-1][1].model.save(saveto['keras_model'])
        model.steps.pop(-1)
        # ... and use joblib to save the rest of the pipeline
        joblib.dump(model, saveto['sklearn_pipe'])

    # Return scores as well as training time via decorator
    return scores


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline

    from sklearn.feature_extraction.text import TfidfVectorizer
    from reader import PickledReviewsReader
    from am_reader import PickledAmazonReviewsReader
    from transformer import TextNormalizer, GensimDoc2Vectorizer
    from transformer import KeyphraseExtractor, GensimTfidfVectorizer

    # Build a Keras Sequential model for the Pitchfork reviews
    cpath = '../review_corpus_proc'
    mpath = {
        'keras_model'  : 'ktf/keras_nn.h5',
        'sklearn_pipe' : 'ktf/pipeline.pkl'
    }

    pipeline = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', TfidfVectorizer(max_features=N_FEATURES)), # need to control feature count
        # ('vect', GensimDoc2Vectorizer(size=N_FEATURES)), # need to control feature count
        ('nn', KerasClassifier(build_fn=build_nn, # pass but don't call the function!
                               epochs=200,
                               batch_size=128))
    ])

    scores, delta = train_model(cpath, pipeline, saveto=mpath, cv=12)
    for idx, score in enumerate(scores):
        print('Accuracy on slice #{}: {}.'.format((idx+1), score))
    print('Total fit time: {:0.2f} seconds'.format(delta))
    print('Model saved to {}.'.format(list(mpath)))


    # Use the Amazon music & movie review corpus & LSTM instead
    apath = '../am_corpus_proc'
    pipeline = Pipeline([
        ('keyphrases', KeyphraseExtractor(nfeatures=N_FEATURES,
                                          doclen=DOC_LEN)),
        ('nn', KerasClassifier(build_fn=build_lstm,
                               epochs=4,
                               batch_size=128))
    ])
    scores, delta = train_model(apath, pipeline, cv=5)
    print('Mean score: {}'.format(np.mean(scores)))
    print('Total fit time: {:0.2f} seconds'.format(delta))
