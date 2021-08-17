import spacy
import pprint
import argparse
from nltk.tree import Tree
from nltk.parse.stanford import StanfordParser


##########################################################################
# Spacy/NLTK
##########################################################################

# Required: python -m spacy download en
spacy_nlp = spacy.load("en")

def spacy_tree(sent):
    """
    Get the SpaCy dependency tree structure
    :param sent: string
    :return: None
    """
    doc = spacy_nlp(sent)
    pprint.pprint(doc.print_tree())

def nltk_spacy_tree(sent):
    """
    Visually inspect the SpaCy dependency tree with nltk.tree
    :param sent: string
    :return: None
    """
    doc = spacy_nlp(sent)
    def token_format(token):
        return "_".join([token.orth_, token.tag_, token.dep_])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(token_format(node),[to_nltk_tree(child) for child in node.children])
        else:
            return token_format(node)

    tree = [to_nltk_tree(sent.root) for sent in doc.sents]
    tree[0].draw()

def id_measure_quest(sent):
    """
    Hacky failed attempt to identify questions with SpaCy.
    If id is successful, prints out conversion chart

    :param sent: string
    :return: None
    """
    doc = spacy_nlp(sent)
    for sent in doc.sents:
        for token in sent:
            if token.tag_ in 'WRB':
                if token.nbor().tag_ == 'JJ':
                    if token.nbor().nbor().tag_ in ('NN', 'NNS'):
                        print(conversion_chart)


##########################################################################
# Stanford/NLTK
##########################################################################
# Required: download Stanford jar dependencies
# https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk
stanford_parser = StanfordParser(
    model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
)

def nltk_stanford_parse(sent):
    """
    Use Stanford pretrained model to extract dependency tree
    for use by other methods
    :param sent: str
    :return: list of trees
    """
    parse = stanford_parser.raw_parse(sent)
    return list(parse)

def nltk_stanford_tree(sent):
    """
    Visually inspect the Stanford dependency tree as an image
    :param sent: str
    :return: NLTK image
    """
    parse = stanford_parser.raw_parse(sent)
    tree = list(parse)
    tree[0].draw()


def question_type(tree):
    """
    Try to identify whether the question is about measurements,
    recipes, or not a question.
    :param tree: list of trees produced by nltk_stanford_parse
    :return: str response type
    """
    noun_tags = ["NNS", "NN", "NNP", "NNPS"]
    nouns = [token for token, tag in tree[0].pos() if tag in noun_tags]
    for sent in tree[0].subtrees():
        # Find direct questions introduced by wh-word or -phrase.
        if sent.label() == "SBARQ":
            for clause in sent.subtrees():
                # Find wh-adjective and wh-adverb phrases
                if clause.label() in ("WHADJP", "WRB"):
                    for token, tag in clause.pos():
                        if token == "How":
                            return ("quantity", nouns)
                # Find wh-noun phrases
                elif clause.label() == "WP":
                    # Use pre-trained clusters to return recipes
                    return ("recipe", nouns)

    # Todo: try to be conversational using our n-gram language generator?
    return ("default", nouns)



if __name__ == '__main__':
    # Do some command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--parser', choices={'nltk', 'syntaxnet'},
        default='nltk', help='select parser to use',
    )
    parser.add_argument('sents', nargs="+", help='sentences to parse, surround by quotes')
    args = parser.parse_args()
    # print(args.sents)

    if args.parser == 'syntaxnet':
        raise NotImplementedError("not done yet")
    elif args.parser == 'nltk':
        for sent in args.sents:
            ## Visualize the trees
            # nltk_spacy_tree(sent) # "Dependency" tree
            # nltk_stanford_tree(sent) # "Constituency" tree

            # Get Stanford dependency tree as a list
            parse = nltk_stanford_parse(sent)
            print(question_type(parse))

    else:
        raise TypeError("'{}' is not a valid parser".format(args.parser))
