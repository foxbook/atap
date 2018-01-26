#!/usr/bin/env python3

import heapq
import collections
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from operator import itemgetter

from entities import pairs
from reader import PickledCorpusReader

def graph(corpus):

    # Create an undirected graph
    G = nx.Graph(name="Baleen Entity Graph")

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

    # Add edges for each document-entities and between pairs
    for idx, doc in enumerate(corpus.docs()):
        fileid = corpus.fileids()[idx]
        for pair in pairs(doc):
            # NOTE: each pair is a tuple with (entity,tag)
            # here I'm adding only the entity to the graph,
            # though it might be interesting to add the tags
            # so we can filter the graph by entity type...
            # G.add_edge(fileid, pair[0][0])
            # G.add_edge(fileid, pair[1][0])
            # Now add edges between entity pairs with a weight
            # of 1 for every document they co-appear in
            if (pair[0][0], pair[1][0]) in G.edges():
                G.edges[(pair[0][0], pair[1][0])]['weight'] += 1
            else:
                G.add_edge(pair[0][0], pair[1][0], weight=1)

    return G


def nbest_centrality(G, metric, n=10, attr="centrality", **kwargs):
    # Compute the centrality scores for each vertex
    scores = metric(G, **kwargs)

    # Set the score as a property on each node
    nx.set_node_attributes(G, name=attr, values=scores)

    # Find the top n scores and print them along with their index
    topn = heapq.nlargest(n, scores.items(), key=itemgetter(1))
    for idx, item in enumerate(topn):
        print("{}. {}: {:0.4f}".format(idx + 1, *item))

    return G

if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')
    G = graph(corpus)

    # # Write the graph to disk, if needed
    # nx.write_graphml(G, "entities.graphml")

    # # Get summary stats for the full graph
    # print(nx.info(G))

    # # find the most central entities in the social network
    # print("Degree centrality")
    # nbest_centrality(G, nx.degree_centrality)
    # print("Betweenness centrality")
    # nbest_centrality(G, nx.betweenness_centrality, 10, "betweenness", normalized=True)

    # # Extract and visualize an ego graph
    # H = nx.ego_graph(G, "Hollywood")
    # edges, weights = zip(*nx.get_edge_attributes(C, "weight").items())
    # pos = nx.spring_layout(C, k=0.3, iterations=40)
    # nx.draw(
    #     C, pos, node_color="skyblue", node_size=20, edgelist=edges,
    #     edge_color=weights, width=0.25, edge_cmap=plt.cm.Pastel2,
    #     with_labels=True, font_size=6, alpha=0.8)
    # plt.show()
    # plt.savefig("atap_ch09_hollywood_entity_graph.png", transparent=True)

    # # Compare centrality measures for an ego graph
    # print("Closeness centrality for Hollywood")
    # nbest_centrality(H, nx.closeness_centrality, 10, "closeness")
    # print("Eigenvector centrality for Hollywood")
    # nbest_centrality(H, nx.eigenvector_centrality_numpy, 10, "eigenvector")
    # print("Pagerank centrality for Hollywood")
    # nbest_centrality(H, nx.pagerank_numpy, 10, "pagerank")
    # print("Katz centrality for Hollywood")
    # nbest_centrality(H, nx.katz_centrality_numpy, 10, "katz")

    # T = nx.ego_graph(G, "Twitter")
    # E = nx.ego_graph(G, "Earth")

    # # Examine degree distributions with histograms
    # sns.distplot(
    #     [G.degree(v) for v in G.nodes()], norm_hist=True
    # )
    # plt.show()
    #
    # sns.distplot(
    #     [H.degree(v) for v in H.nodes()], norm_hist=True
    # )
    # plt.show()
    #
    # sns.distplot(
    #     [T.degree(v) for v in T.nodes()], norm_hist=True
    # )
    # plt.show()
    #
    # sns.distplot(
    #     [E.degree(v) for v in E.nodes()], norm_hist=True
    # )
    # plt.show()
    #
    # print("Baleen Entity Graph")
    # print("Transitivity: {}".format(nx.transitivity(G)))
    # print("Average clustering coefficient: {}".format(nx.average_clustering(G)))
    # print("Number of cliques: {}".format(nx.graph_number_of_cliques(G)))
    #
    # print("Hollywood Ego Graph")
    # print("Transitivity: {}".format(nx.transitivity(H)))
    # print("Average clustering coefficient: {}".format(nx.average_clustering(H)))
    # print("Number of cliques: {}".format(nx.graph_number_of_cliques(H)))
