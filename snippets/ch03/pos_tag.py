from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

def sents(paragraph):
    for sentence in sent_tokenize(paragraph):
        yield sentence

def tokenize(paragraph):
    for sentence in sents(paragraph):
        yield pos_tag(wordpunct_tokenize(sentence))

sample_text = "The old building is scheduled for demolition. The contractors will begin building a new structure next month."
print(list(tokenize(sample_text)))
