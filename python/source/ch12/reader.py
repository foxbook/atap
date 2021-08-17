#!/usr/bin/env python3

import nltk
import pickle
import sqlite3

from nltk.corpus.reader.api import CorpusReader

PKL_PATTERN = r'(?!\.)[\w\s\d\-]+\.pickle'

class SqliteCorpusReader(object):

    def __init__(self, path):
        self._cur = sqlite3.connect(path).cursor()

    def scores(self):
        """
        Returns the review score
        """
        self._cur.execute("SELECT score FROM reviews")
        scores = self._cur.fetchall()
        for score in scores:
            yield score

    def texts(self):
        """
        Returns the full review texts
        """
        self._cur.execute("SELECT content FROM content")
        texts = self._cur.fetchall()
        for text in texts:
            yield text

    def ids(self):
        """
        Returns the review ids
        """
        self._cur.execute("SELECT reviewid FROM content")
        ids = self._cur.fetchall()
        for idx in ids:
            yield idx

    def ids_and_texts(self):
        """
        Returns the review ids
        """
        self._cur.execute("SELECT * FROM content")
        results = self._cur.fetchall()
        for idx,text in results:
            yield idx,text

    def scores_albums_artists_texts(self):
        """
        Returns a generator with each review represented as a
        (score, album name, artist name, review text) tuple
        """
        sql = """
              SELECT S.score, L.label, A.artist, R.content
              FROM [reviews] S
              JOIN labels L ON S.reviewid=L.reviewid
              JOIN artists A on L.reviewid=A.reviewid
              JOIN content R ON A.reviewid=R.reviewid
              """
        self._cur.execute(sql)
        results = self._cur.fetchall()
        for score,album,band,text in results:
            yield (score,album,band,text)

    def albums(self):
        """
        Returns the names of albums being reviewed
        """
        self._cur.execute("SELECT * FROM labels")
        albums = self._cur.fetchall()
        for idx,album in albums:
            yield idx,album

    def artists(self):
        """
        Returns the name of the artist being reviewed
        """
        self._cur.execute("SELECT * FROM artists")
        artists = self._cur.fetchall()
        for idx,artist in artists:
            yield idx,artist

    def genres(self):
        """
        Returns the music genre of each review
        """
        self._cur.execute("SELECT * FROM genres")
        genres = self._cur.fetchall()
        for idx,genre in genres:
            yield idx,genre

    def years(self):
        """
        Returns the publication year of each review

        Note: There are many missing values
        """
        self._cur.execute("SELECT * FROM years")
        years = self._cur.fetchall()
        for idx,year in years:
            yield idx,year

    def paras(self):
        """
        Returns a generator of paragraphs.
        """
        for text in self.texts():
            for paragraph in text:
                yield paragraph

    def sents(self):
        """
        Returns a generator of sentences.
        """
        for para in self.paras():
            for sentence in nltk.sent_tokenize(para):
                yield sentence

    def words(self):
        """
        Returns a generator of words.
        """
        for sent in self.sents():
            for word in nltk.wordpunct_tokenize(sent):
                yield word

    def tagged_tokens(self):
        for sent in self.sents():
            for word in nltk.wordpunct_tokenize(sent):
                yield nltk.pos_tag(word)



class PickledReviewsReader(CorpusReader):
    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader
        """
        CorpusReader.__init__(self, root, fileids, **kwargs)

    def texts_scores(self, fileids=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the SqliteCorpusReader, this uses a generator
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

    def paras(self, fileids=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for review in self.reviews(fileids):
            for paragraph in review:
                yield paragraph

    def sents(self, fileids=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(fileids):
            for sentence in paragraph:
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
    # Download the data from https://www.kaggle.com/nolanbconaway/pitchfork-data/data
    # preprocess by running preprocess.py to produce pickled corpus
    reader = PickledReviewsReader('../review_corpus_proc')
    print(len(list(reader.reviews())))
