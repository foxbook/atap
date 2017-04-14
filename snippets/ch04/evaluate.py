from build import models, reader
from build import labels as categories
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report

docs = reader.fileids(categories=categories)
labels = [reader.categories(fileids=[fid])[0] for fid in docs]

train_docs, test_docs, train_labels, test_labels = tts(docs, labels, test_size=0.2)

def get_docs(fids):
    for fid in fids:
        yield list(reader.docs(fileids=[fid]))

sgd = models[3]
nby = models[4]


sgd.fit(get_docs(train_docs), train_labels)
y_pred = sgd.predict(get_docs(test_docs))

print(classification_report(test_labels, y_pred, labels=categories))


import nltk

def preprocess(text):
    return [
        [
            list(nltk.pos_tag(nltk.word_tokenize(sent)))
            for sent in nltk.sent_tokenize(para)
        ] for para in text.split("\n\n")
    ]


doc = preprocess("""
Last summer, two defensemen from opposing conferences with distinct styles of play and contrasting personalities were forever placed in the same breath, their destinies intertwined by a trade.

The Nashville Predators sent Shea Weber, their cornerstone, to the Montreal Canadiens for P. K. Subban, who had become tremendously popular in Montreal and throughout the league. Subban, 27, won a Norris Trophy as the league’s top defenseman in 2013. Weber, 31, had been a three-time finalist for the award.

“Sometimes you forget that superstars get traded,” Anaheim Ducks defenseman Cam Fowler said. “Obviously, what P. K. meant to Montreal and the impact that he had on that city, it was hard for them to let him go. The same with Shea, who was their captain for years.”

Weber and Subban were together again at last weekend’s All-Star three-on-three tournament. Weber’s 31 points in 50 games for the first-place Canadiens, and his plus-18 rating, made him an obvious selection. Subban was voted in as a team captain by the fans despite a mixed first half of the season. He posted only 18 points and missed 16 games for the Predators, who are in third place in the Central Division.
""")

# print(doc[0][0])
print(sgd.predict(doc[0][0]))
