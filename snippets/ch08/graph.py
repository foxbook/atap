#!/usr/bin/env python3

import networkx as nx

from reader import PickledCorpusReader
from candidates import scored_document_phrases


def graph(corpus):

    # Create an undirected graph
    G = nx.Graph(name="Baleen Keyphrase Graph")

    # Create category, feed, and document nodes
    G.add_nodes_from(corpus.categories(), type='category')
    G.add_nodes_from([feed['title'] for feed in corpus.feeds()], type='feed')
    G.add_nodes_from(corpus.fileids(), type='document')

    # Create feed-category edges
    G.add_edges_from([
        (feed['title'], feed['category']) for feed in corpus.feeds()
    ])

    # Create document-category edges
    G.add_edges_from([
        (fileid, corpus.categories(fileids=fileid)[0])
        for fileid in corpus.fileids()
    ])

    # Perform keyphrase extraction using TF-IDF
    scores = scored_document_phrases([
        corpus.sents(fileids=[fileid]) for fileid in corpus.fileids()
    ], segmented=True)

    # add keyphrase-document edges
    for idx, doc in enumerate(scores):
        fileid = corpus.fileids()[idx]

        for phrase, tfidf in doc:
            G.add_node(phrase, type='keyphrase')
            G.add_edge(fileid, phrase, weight=tfidf)

    return G


if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')
    G = graph(corpus)
    nx.write_graphml(G, "phrases.graphml")

    print(nx.info(G))
