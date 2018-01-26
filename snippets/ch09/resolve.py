import networkx as nx
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

from itertools import combinations


def build_graph(edges_and_props, name='Entity Graph'):
    G = nx.Graph(name=name)

    for n1,n2 in edges_and_props:
        G.add_node(n1[0], type=n1[1])
        G.add_node(n2[0], type=n2[1])
        G.add_edge(n1[0], n2[0])

    return G

def pairwise_comparisons(G):
    """
    Produces a generator of pairs of nodes.
    """
    return combinations(G.nodes(), 2)


def edge_blocked_comparisons(G):
    """
    A generator of pairwise comparisons, that highlights comparisons
    between nodes that have an edge to the same entity.
    """
    for n1, n2 in pairwise_comparisons(G):
        hood1 = frozenset(G.neighbors(n1))
        hood2 = frozenset(G.neighbors(n2))
        if hood1 & hood2:
            yield n1,n2


def similarity(n1, n2):
    """
    Returns the mean of the partial_ratio score for each field in the two
    entities. Note that if they don't have fields that match, the score will
    be zero.
    """
    scores = [
        fuzz.partial_ratio(n1, n2),
        fuzz.partial_ratio(G.node[n1]['type'], G.node[n2]['type'])
    ]

    return float(sum(s for s in scores)) / float(len(scores))


def fuzzy_blocked_comparisons(G, threshold=65):
    """
    A generator of pairwise comparisons, that highlights comparisons between
    nodes that have an edge to the same entity, but filters out comparisons
    if the similarity of n1 and n2 is below the threshold.
    """
    for n1, n2 in pairwise_comparisons(G):
        hood1 = frozenset(G.neighbors(n1))
        hood2 = frozenset(G.neighbors(n2))
        if hood1 & hood2:
            if similarity(n1, n2) > threshold:
                yield n1,n2

def info(G):
    """
    Wrapper for nx.info with some other helpers.
    """
    pairwise = len(list(pairwise_comparisons(G)))
    edge_blocked = len(list(edge_blocked_comparisons(G)))
    fuzz_blocked = len(list(fuzzy_blocked_comparisons(G)))

    output = [""]
    output.append("Number of Pairwise Comparisons: {}".format(pairwise))
    output.append("Number of Edge Blocked Comparisons: {}".format(edge_blocked))
    output.append("Number of Fuzzy Blocked Comparisons: {}".format(fuzz_blocked))

    return nx.info(G) + "\n".join(output)

if __name__ == '__main__':

    hilton_edges = [
        (('Hilton', 'ORG'), ('Hilton Hotels', 'ORG')),
        (('Hilton', 'ORG'), ('Hilton Hotels and Resorts', 'ORG')),
        (('Hilton', 'ORG'), ('Hilton Hotels', 'ORG')),
        (('Hilton', 'ORG'), ('Paris', 'GPE')),
        (('Hilton', 'ORG'), ('Hiltons', 'ORG')),
        (('Paris', 'GPE'), ('Charles de Gaulle', 'FACILITY')),
        (('Paris', 'GPE'), ('Parisian Hiltons', 'ORG')),
        (('Paris', 'GPE'), ('Hilton Paris Opera', 'FACILITY')),
        (('Paris', 'GPE'), ('Paris, France', 'GPE')),
        (('Hiltons', 'ORG'), ('Kathy Hilton', 'PERSON')),
        (('Hiltons', 'ORG'), ('Richard Hilton', 'PERSON')),
        (('Hiltons', 'ORG'), ('Rick Hilton', 'PERSON')),
        (('Hiltons', 'ORG'), ('Paris Hilton', 'PERSON')),
        (('Paris Hilton', 'PERSON'), ('Paris Whitney Hilton', 'PERSON')),
        (('Paris Hilton', 'PERSON'), ('Elliot Mintz', 'PERSON')),
        (('Conrad Hilton', 'PERSON'), ('Conrad', 'ORG')),
        (('Conrad Hilton', 'PERSON'), ('Conrad Nicholson Hilton', 'PERSON')),
        (('Conrad', 'ORG'), ('Conrad Nicholson Hilton', 'PERSON'))
    ]

    G = build_graph(hilton_edges, name="Hilton Family")


    pos = nx.spring_layout(G, k=0.25, iterations=30)
    nx.draw(
        G, pos, node_color="lightgreen", node_size=20,
        edge_color='darkgreen', width=0.5, with_labels=True,
        font_size=6, alpha=0.8
    )
    plt.show()

    print(info(G))