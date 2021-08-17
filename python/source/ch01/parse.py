import nltk

grammar = nltk.CFG.fromstring("""
S -> NP VP PUNCT
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'the'
V -> 'put'
N -> 'book' | 'box' | 'table'
P -> 'in' | 'on'
PUNCT -> '.'
""")

def parse(sent):
    parser = nltk.ChartParser(grammar)
    tokens = nltk.wordpunct_tokenize(sent)
    return parser.parse(tokens)

if __name__ == '__main__':
    for tree in parse("I put the book in the box on the table."):
        tree.draw()
