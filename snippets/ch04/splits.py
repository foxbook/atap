from sklearn.model_selection import train_test_split as tts
from reader import PickledCorpusReader

reader = PickledCorpusReader('../corpus')

labels = ["books", "cinema", "cooking", "gaming", "sports", "tech"]
docs = reader.fileids(categories=labels)
X = list(reader.docs(fileids=docs))
y = [reader.categories(fileids=[fileid])[0] for fileid in docs]
