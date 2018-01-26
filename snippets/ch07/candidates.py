#!/usr/bin/env python3

import nltk
import gensim

from itertools import groupby
from nltk.chunk import tree2conlltags
from nltk.chunk.regexp import RegexpParser

from unicodedata import category as unicat

from tabulate import tabulate

GRAMMAR   = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
GOODTAGS  = set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])


def normalize(sent):

    is_punct = lambda word: all(unicat(c).startswith('P') for c in word)

    # Removes punctuation from a tokenized/tagged sentence and lowercases.
    sent = filter(lambda t: not is_punct(t[0]), sent)
    sent = map(lambda t: (t[0].lower(), t[1]), sent)
    return list(sent)


def extract_candidate_terms(sents, tags=GOODTAGS, tagged=False):

    for sent in sents:
        # Tokenize and tag sentences if necessary
        if not tagged:
            sent = nltk.pos_tag(nltk.word_tokenize(sent))

        # Identify only good terms by the set of passed in tags
        for term, tag in normalize(sent):
            if tag in tags:
                yield term.lower()


def extract_candidate_phrases(sents, grammar=GRAMMAR, tagged=False):

    # Create the chunker that uses our grammar
    chunker = RegexpParser(grammar)

    for sent in sents:
        # Tokenize and tag sentences if necessary
        if not tagged:
            sent = nltk.pos_tag(nltk.word_tokenize(sent))

        # Parse the sentence, converting the parse tree into a tagged sequence
        sent = normalize(sent)
        if not sent: continue
        chunks = tree2conlltags(chunker.parse(sent))

        # Extract phrases and rejoin them with space
        phrases = [
            " ".join(word for word, pos, chunk in group).lower()
            for key, group in groupby(
                chunks, lambda term: term[-1] != 'O'
            ) if key
        ]

        for phrase in phrases:
            yield phrase


def scored_document_phrases(documents, segmented=True):

    # If documents are not segmented and tagged, do so.
    if not segmented:
        documents = [
            nltk.sent_tokenize(document)
            for document in documents
        ]

    # Compose the documents as a list of their keyphrases
    documents = [
        list(extract_candidate_phrases(document, tagged=segmented))
        for document in documents
    ]

    # Create a lexicon of candidate phrases
    lexicon = gensim.corpora.Dictionary(documents)

    # Vectorize the documents by phrases for scoring
    vectors = [
        lexicon.doc2bow(document)
        for document in documents
    ]

    # Create the TF-IDF Model and compute the scores
    model = gensim.models.TfidfModel(vectors)
    scores = model[vectors]

    for doc in scores:
        yield [
            (lexicon[vec], score) for vec, score in doc
        ]


if __name__ == '__main__':

    import heapq

    from reader import PickledCorpusReader
    from collections import Counter

    corpus = PickledCorpusReader('../corpus')
    scores = scored_document_phrases([
        list(corpus.sents(fileids=fileid)) for fileid in corpus.fileids(categories=["politics", "news"])
    ], True)
    tfidfs = Counter()

    for phrases in scores:
        for phrase, score in phrases:
            tfidfs[phrase] += score

    print(
        tabulate(tfidfs.most_common(50), headers=["keyphrase", "cumulative tfidf"])
    )
