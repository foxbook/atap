
from reader import PickledCorpusReader

reader = PickledCorpusReader('../corpus')

for category in reader.categories():

    n_docs = len(reader.fileids(categories=[category]))
    n_words = sum(1 for word in reader.words(categories=[category]))

    print("- '{}' contains {:,} docs and {:,} words".format(category, n_docs, n_words))
