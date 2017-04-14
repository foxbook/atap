#!/usr/bin/env python3

import nltk
import string

# Tokenization function
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


# The corpus object
corpus = [
    "The elephant sneezed at the sight of potatoes.",
    "Bats can see via echolocation. See the bat sight sneeze!",
    "Wondering, she opened the door to the studio.",
]

def nltk_frequency_vectorize(corpus):

    # The NLTK frequency vectorize method
    from collections import defaultdict

    def vectorize(doc):
        features = defaultdict(int)

        for token in tokenize(doc):
            features[token] += 1

        return features

    return map(vectorize, corpus)


def sklearn_frequency_vectorize(corpus):
    # The Scikit-Learn frequency vectorize method
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)


def gensim_frequency_vectorize(corpus):
    # The Gensim frequency vectorize method
    import gensim

    id2word = gensim.corpora.Dictionary([
        list(vectorize(doc)) for doc in corpus]
    )

    return [
        id2word.doc2bow(doc) for doc in corpus
    ]


def nltk_one_hot_vectorize(corpus):
    # The NLTK one hot vectorize method
    def vectorize(doc):
        return {
            token: True
            for token in tokenize(doc)
        }

    return map(vectorize, corpus)


def sklearn_one_hot_vectorize(corpus):
    # The Sklearn one hot vectorize method

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import Binarizer

    freq    = CountVectorizer()
    vectors = freq.fit_transform(corpus)

    print(len(vectors.toarray()[0]))

    onehot  = Binarizer()
    vectors = onehot.fit_transform(vectors.toarray())

    print(len(vectors[0]))


def gensim_one_hot_vectorize(corpus):
    # The Gensim one hot vectorize method
    import gensim
    import numpy as np

    corpus  = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus)

    corpus  = np.array([
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in corpus
    ])

    return corpus


def nltk_tfidf_vectorize(corpus):

    from nltk.text import TextCollection

    corpus = [list(tokenize(doc)) for doc in corpus]
    texts = TextCollection(corpus)

    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }


def sklearn_tfidf_vectorize(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)


def gensim_tfidf_vectorize(corpus):
    import gensim

    corpus  = [list(tokenize(doc)) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(corpus)

    tfidf   = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
    vectors = [tfidf[lexicon.doc2bow(vector)] for vector in corpus]

    lexicon.save_as_text('test.txt')
    tfidf.save('tfidf.pkl')

    return vectors


def gensim_doc2vec_vectorize(corpus):
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    corpus = [list(tokenize(doc)) for doc in corpus]
    docs   = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
    ]
    model = Doc2Vec(docs, size=5, min_count=0)
    return model.docvecs


print(gensim_doc2vec_vectorize(corpus)[0])
