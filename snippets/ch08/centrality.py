#!/usr/bin/env python

import heapq
import networkx as nx

from operator import itemgetter

def nbest_centrality(G, metric, n=10, attr="centrality", **kwargs):
    # Compute the centrality scores for each vertex
    scores = metric(G, **kwargs)

    # Set the score as a property on each node
    nx.set_node_attributes(G, attr, scores)

    # Filter scores (do not include in book)
    ntypes = nx.get_node_attributes(G, 'type')
    phrases = [
        item for item in scores.items()
        if ntypes.get(item[0], None) == "keyphrase"
    ]

    # Find the top n scores and print them along with their index
    topn = heapq.nlargest(n, phrases, key=itemgetter(1))
    for idx, item in enumerate(topn):
        print("{}. {}: {:0.4f}".format(idx+1, *item))

    return G


if __name__ == '__main__':
    G = nx.read_graphml("phrases.graphml")
    nbest_centrality(G, nx.betweenness_centrality, 25, "degree", normalized=True)
