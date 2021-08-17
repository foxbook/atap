#!/usr/bin/env python3

import os
import nltk
import gensim
import numpy as np
import unicodedata

from itertools import groupby
from unicodedata import category as unicat

from nltk.corpus import wordnet as wn
from nltk.chunk import tree2conlltags
from nltk.probability import FreqDist
from nltk.chunk.regexp import RegexpParser
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing import sequence

from gensim.matutils import sparse2full, full2sparse, full2sparse_clipped, scipy2scipy_clipped
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
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
            for sentence in document
            for (token, tag) in sentence
            if not self.is_punct(token)
               and not self.is_stopword(token)
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
        return [
            ' '.join(self.normalize(doc)) for doc in documents
        ]


class GensimDoc2Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, size=5, min_count=3):
        """
        gensim_doc2vec_vectorize
        """
        self.size = size
        self.min_count = min_count

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        docs = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(documents)
        ]
        model = Doc2Vec(docs, size=self.size, min_count=self.min_count)
        return np.array(list(model.docvecs))

class GensimTfidfVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, nfeatures=100, tofull=False):
        """
        Pass in a directory that holds the lexicon in corpus.dict and the
        TFIDF model in tfidf.model (for now).

        Set tofull = True if the next thing is a Scikit-Learn estimator
        otherwise keep False if the next thing is a Gensim model.
        """
        self._lexicon_path = "lexigram.dict"
        self._tfidf_path = "tfidf.model"
        self.nfeatures = nfeatures
        self.lexicon = None
        self.tfidf = None
        self.tofull = tofull

        self.load()

    def load(self):
        if os.path.exists(self._lexicon_path):
            self.lexicon = gensim.corpora.Dictionary.load(self._lexicon_path)

        if os.path.exists(self._tfidf_path):
            self.tfidf = gensim.models.TfidfModel().load(self._tfidf_path)

    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.tfidf.save(self._tfidf_path)

    def fit(self, documents, labels=None):
        self.lexicon = gensim.corpora.Dictionary(documents, prune_at=self.nfeatures)
        self.lexicon.filter_extremes(keep_n=self.nfeatures)
        self.lexicon.compactify()
        self.tfidf = gensim.models.TfidfModel(
            [self.lexicon.doc2bow(doc) for doc in documents],
            id2word=self.lexicon
        )
        self.save()
        return self

    def transform(self, documents):
        def generator():
            for document in documents:
                vec = self.tfidf[self.lexicon.doc2bow(document)]
                if self.tofull:
                    yield sparse2full(vec, len(self.lexicon))
                else:
                    yield vec
        return np.array(list(generator()))


class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    """
    Extract adverbial and adjective phrases, and transform
    documents into lists of these keyphrases, with a total
    keyphrase lexicon limited by the nfeatures parameter
    and a document length limited/padded to doclen
    """
    def __init__(self, nfeatures=100000, doclen=60):
        self.grammar = r'KT: {(<RB.> <JJ.*>|<VB.*>|<RB.*>)|(<JJ> <NN.*>)}'
        # self.grammar = r'KT: {(<RB.*> <VB.>|<RB.>|<JJ.> <NN.*>)}'
        # self.grammar = r'KT: {<RB.>|<JJ.>}'
        self.chunker = RegexpParser(self.grammar)
        self.nfeatures = nfeatures
        self.doclen = doclen

    def normalize(self, sent):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        is_punct = lambda word: all(unicat(c).startswith('P') for c in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_candidate_phrases(self, sents):
        """
        For a document, parse sentences using our chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Extract phrases, rejoin with a space, and yield the document
        represented as a list of it's keyphrases.
        """
        for sent in sents:
            sent = self.normalize(sent)
            if not sent: continue
            chunks = tree2conlltags(self.chunker.parse(sent))
            phrases = [
                " ".join(word for word, pos, chunk in group).lower()
                for key, group in groupby(
                    chunks, lambda term: term[-1] != 'O'
                ) if key
            ]
            for phrase in phrases:
                yield phrase

    def fit(self, documents, y=None):
        return self

    def get_lexicon(self, keydocs):
        """
        Build a lexicon of size nfeatures
        """
        keyphrases = [keyphrase for doc in keydocs for keyphrase in doc]
        fdist = FreqDist(keyphrases)
        counts = fdist.most_common(self.nfeatures)
        lexicon = [phrase for phrase, count in counts]
        return {phrase: idx+1 for idx, phrase in enumerate(lexicon)}

    def clip(self, keydoc, lexicon):
        """
        Remove keyphrases from documents that aren't in the lexicon
        """
        return [lexicon[keyphrase] for keyphrase in keydoc
                if keyphrase in lexicon.keys()]

    def transform(self, documents):
        docs = [list(self.extract_candidate_phrases(doc)) for doc in documents]
        lexicon = self.get_lexicon(docs)
        clipped = [list(self.clip(doc, lexicon)) for doc in docs]
        return sequence.pad_sequences(clipped, maxlen=self.doclen)


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    from am_reader import PickledAmazonReviewsReader
    
    corpus = PickledAmazonReviewsReader("../am_corpus_proc")
    keydocs = list(KeyphraseExtractor().fit_transform(corpus.reviews()))
    for doc in keydocs:
        print(doc)
