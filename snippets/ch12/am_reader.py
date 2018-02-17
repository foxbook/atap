#!/usr/bin/env python3

import os
import json
import nltk
import codecs
import pickle

from nltk.corpus.reader.api import CorpusReader

DOC_PATTERN = r'.*\.json'
PKL_PATTERN = r'.*\.pickle'

class JsonCorpusReader(CorpusReader):

    def __init__(self, root, fileids=DOC_PATTERN, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids):
        """
        Returns a list of fileids.
        """
        return fileids

    def reviews(self, fileids=None):
        """
        Returns the complete text of the JSON document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)

    def texts(self):
        """
        Returns the full review texts
        """
        for review in self.reviews():
            yield review["reviewText"]

    def scores(self):
        """
        Returns the review scores
        """
        for review in self.reviews():
            yield review["overall"]

    def ids(self):
        """
        Returns the review ids
        """
        for review in self.reviews():
            yield review["unixReviewTime"]

    def ids_scores_texts(self):
        """
        Returns the review ids, scores & texts
        """
        for review in self.reviews():
            yield (review["unixReviewTime"], review["overall"], review["reviewText"])

    def sents(self):
        """
        Returns a generator of sentences.
        """
        for text in self.texts():
            for sentence in nltk.sent_tokenize(text):
                yield sentence

    def words(self):
        """
        Returns a generator of words.
        """
        for sent in self.sents():
            for word in nltk.wordpunct_tokenize(sent):
                yield word

    def tagged_sents(self):
        for sent in self.sents():
            yield nltk.pos_tag(nltk.wordpunct_tokenize(sent))



class PickledAmazonReviewsReader(CorpusReader):
    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader
        """
        CorpusReader.__init__(self, root, fileids, **kwargs)

    def texts_scores(self, fileids=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the JsonCorpusReader, this uses a generator
        to achieve memory safe iteration.
        """
        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def reviews(self, fileids=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for text,score in self.texts_scores(fileids):
            yield text

    def scores(self, fileids=None):
        """
        Return the scores
        """
        for text,score in self.texts_scores(fileids):
            yield score

    def sents(self, fileids=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for review in self.reviews(fileids):
            for sentence in review:
                yield sentence

    def tagged(self, fileids=None):
        for sent in self.sents(fileids):
            for token in sent:
                yield token

    def words(self, fileids=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for token in self.tagged(fileids):
            yield token[0]




if __name__ == '__main__':

    # # One-time run to convert malformed single JSON to separate JSON files
    # target = "../am_corpus"
    # for review in open('../reviews_Movies_and_TV_5.json', 'rb'):
    #     data = json.loads(review)
    #     fname = data['reviewerID']+'_'+str(data['unixReviewTime']) + '.json'
    #     abspath = os.path.normpath(os.path.join(target, fname))
    #     parent = os.path.dirname(abspath)
    #     if not os.path.exists(parent):
    #         os.makedirs(parent)
    #     with open(abspath, 'w') as outfile:
    #         json.dump(data, outfile)

    corpus = JsonCorpusReader('../am_corpus')
    print(list(corpus.reviews(fileids="A1A0AAJX62NHD8_1363219200.json")))
    # corpus = PickledAmazonReviewsReader('../am_corpus_proc')
    # print(len(list(corpus.reviews())))
