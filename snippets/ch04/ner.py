import os
import nltk
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

path = os.path.join(os.getcwd(), "debates")

DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
CAT_PATTERN = r'([\w_\s]+)/.*'

corpus = CategorizedPlaintextCorpusReader(
    path, DOC_PATTERN, cat_pattern=CAT_PATTERN)


def tag_corpus(corpus):
    return [nltk.pos_tag(sent) for sent in corpus.sents()]

tagged_corpus = tag_corpus(corpus)

import spacy

nlp = spacy.load('en')


def spacy_ner(tokenized_sent):
    doc = nlp(' '.join(tokenized_sent))
    for ent in doc.ents:
        return ent.text, ent.label_

spacy_named_entities = [spacy_ner(sent) for sent in corpus.sents()]
spacy_named_entities = [
    entity for entity in spacy_named_entities if entity is not None]

spacy_named_entities = list(set(spacy_named_entities))
print(spacy_named_entities)
