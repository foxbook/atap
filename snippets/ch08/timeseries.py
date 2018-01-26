import os
import csv
import sys
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from operator import itemgetter
from datetime import date, datetime
from collections import Counter, defaultdict

from normalize import TextNormalizer

# Constants
BASE = os.path.dirname(__file__)
PUBDATES = os.path.join(BASE, "pubdates.csv") # mapping of docid to pubdate
SERIES = os.path.join(BASE, "wordseries.json") # word:date count
DATEFMT = "%Y-%m-%d"


def docid(fileid):
    """
    Returns the docid parsed from the file id
    """
    fname = os.path.basename(fileid)
    return os.path.splitext(fname)[0]


def parse_date(ts):
    """
    Helper function to handle weird mongo datetime output.
    """
    return datetime.strptime(ts.split()[0], DATEFMT)


def load_pubdates(fileids, path=PUBDATES):
    fileids = frozenset([docid(fileid) for fileid in fileids])

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header
        return {
            row[0]: parse_date(row[1])
            for row in reader if row[0] in fileids and row[1]
        }


class WordSeries(object):

    @classmethod
    def load(klass, path):
        """
        Load the word series from disk.
        """
        obj = klass()

        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                word = data['word']
                for dt, val in data['series'].items():
                    dt = datetime.strptime(dt, DATEFMT)
                    obj.words[word][dt] = val

        return obj

    def __init__(self):
        # a map of token -> date -> count
        self.words = defaultdict(Counter)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, key):
        if key not in self.words:
            raise KeyError("'{}' not in word list".format(key))

        # Return a timeseries for the specified key.
        values = []
        dates = []

        for date, value in sorted(self.words[key].items(), key=itemgetter(0)):
            values.append(value)
            dates.append(date)

        return pd.Series(values, index=dates, name=key)

    def read(self, corpus, pubdates=PUBDATES):
        """
        Creates a time series for each unique word in the corpus, normalizing
        the words and filtering out stopwords, etc.
        """
        normalize = TextNormalizer().normalize
        pubdates = load_pubdates(corpus.fileids(), pubdates)

        # count all tokens over time
        for fileid, doc in zip(corpus.fileids(), corpus.docs()):
            fileid = docid(fileid)
            pubdate = pubdates.get(fileid, None)
            if pubdate is None:
                continue

            for token in normalize(doc):
                self.words[token][pubdate] += 1 # self.words hold token to date to count

    def dump(self, path):
        """
        Dump the word series to disk an easily parseable manner (jsonl).
        """
        with open(path, 'w') as f:
            for word, counts in self.words.items():
                obj = {
                    'word': word,
                    'series': {
                        dt.strftime(DATEFMT): val
                        for dt, val in counts.items()
                    }
                }
                f.write(json.dumps(obj))
                f.write("\n")

    def plot(self, *terms):
        """
        Plot the word series for each term in the terms list
        """
        fig, ax = plt.subplots(figsize=(9,6))
        for term in terms:
            self[term].plot(ax=ax)

        ax.set_title("Token Frequency over Time")
        ax.set_ylabel("word count")
        ax.set_xlabel("publication date")
        ax.set_xlim(('2016-02-29','2016-05-25'))
        ax.legend()
        return ax


if __name__ == '__main__':

    if not os.path.exists(SERIES):
        # Build the word series and dump it to disk.

        from reader import PickledCorpusReader
        corpus = PickledCorpusReader('../corpus')

        series = WordSeries()
        series.read(corpus)
        series.dump(SERIES)
        print("wrote {} word series to {}".format(len(series), SERIES))

    sns.set_context('paper')
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    # Load the corpus from disk
    series = WordSeries.load(SERIES)
    series.plot(*sys.argv[1:])
    plt.show()

    # TODO: find an interesting time frame to sample
    # Pop culture craze like Stranger things or Game of thrones?
