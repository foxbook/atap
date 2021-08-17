## Spark Application - execute with spark-submit

## Imports
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from sklearn.ensemble import AdaBoostClassifier

from sc_vectorization import load_corpus, make_vectorizer


## Module Constants
APP_NAME = "Scikit-Learn Sample Classifier"
CORPUS = "text_corpus/corpus/*/*.txt"


def make_accuracy_closure(model, correct, incorrect):
    # model should be a broadcast variable
    # correct and incorrect should be acculumators
    def inner(rows):
        X = []
        y = []

        for row in rows:
            X.append(row['tfidf'])
            y.append(row['label'])

        yp = model.value.predict(X)
        for yi, ypi in zip(y, yp):
            if yi == ypi:
                correct.add(1)
            else:
                incorrect.add(1)
    return inner


## Main functionality
def main(sc, spark):
    # Load and vectorize the corpus
    corpus = load_corpus(sc, spark)
    vector = make_vectorizer().fit(corpus)
    corpus = vector.transform(corpus)

    # Get the sample from the dataset
    sample = corpus.sample(False, 0.1).collect()
    X = [row['tfidf'] for row in sample]
    y = [row['label'] for row in sample]

    # Train a Scikit-Learn Model
    clf = AdaBoostClassifier()
    clf.fit(X, y)

    # Broadcast the Scikit-Learn Model to the cluster
    clf = sc.broadcast(clf)

    # Create accumulators for correct vs incorrect
    correct = sc.accumulator(0)
    incorrect = sc.accumulator(1)

    # Create the accuracy closure
    accuracy = make_accuracy_closure(clf, incorrect, correct)

    # Compute the number incorrect and correct
    corpus.foreachPartition(accuracy)

    accuracy = float(correct.value) / float(correct.value + incorrect.value)
    print("Global accuracy of model was {}".format(accuracy))


if __name__ == "__main__":
    # Configure Spark
    conf  = SparkConf().setAppName(APP_NAME)
    conf  = conf.setMaster("local[*]")
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Execute Main functionality
    main(sc, spark)
