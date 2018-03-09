from nltk import sent_tokenize

def sents(paragraph):
    for sentence in sent_tokenize(paragraph):
        yield sentence

def parse(textfile):
    with open(textfile, 'r') as f:
        text = f.read()
        return list(sents(text))

print(parse("zen.txt"))
print(parse("rhyme.txt"))
