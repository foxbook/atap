import spacy
import heapq
import itertools
import networkx as nx

from operator import itemgetter

from reader import PickledCorpusReader

nlp = spacy.load('en')

GOOD_ENTS = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC',
             'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']

def entities(sent):
    doc = nlp(sent)
    for ent in doc.ents:
        #  filter out non social entities
        if ent.label_ in GOOD_ENTS:
            return ent.text, ent.label_
        else:
            pass

def pairs(doc):
    candidates = [
        entities(' '.join(word for word, tag in sent))
        for para in doc for sent in para
    ]

    doc_entities = [
        entity for entity in candidates if entity is not None
    ]

    return list(itertools.permutations(set(doc_entities), 2))


def graph(docs):
    G = nx.Graph()
    for doc in docs:
        for pair in pairs(doc):
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
    corpus = PickledCorpusReader("../corpus")
    docs = corpus.docs()
    G = graph(docs)

