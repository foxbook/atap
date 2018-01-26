import nltk
import spacy
import pprint
import argparse

from spacy.en import English
from spacy.pipeline import DependencyParser


def spacy_parse(sents):
    # Required: python -m spacy download en
    nlp = spacy.load("en")

    for sent in sents:
        doc = nlp(sent)
        print(doc.print_tree())


if __name__ == '__main__':
    # Do some command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--parser', choices={'nltk', 'spacy', 'syntaxnet'},
        default='spacy', help='select parser to use',
    )
    parser.add_argument('sents', nargs="+", help='sentences to parse, surround by quotes')
    args = parser.parse_args()

    for sent in args.sents:
        pprint.pprint(sent)

    # Now start messing around
    if args.parser == 'nltk':
        raise NotImplementedError("not done yet")
    elif args.parser == 'spacy':
        spacy_parse(args.sents)

    elif args.parser == 'syntaxnet':
        raise NotImplementedError("not done yet")
    else:
        raise TypeError("'{}' is not a valid parser".format(args.parser))
