## Spark Application - execute with spark-submit

## Imports
import os
import nltk

from tabulate import tabulate
from functools import partial
from operator import itemgetter, add
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

tabulate = partial(tabulate, tablefmt="simple", headers="firstrow")

## Module Constants
APP_NAME = "Label and Bigram Count"
CORPUS = "text_corpus/corpus/*/*.txt"

## Closure Functions
def parse_label(path):
    # Returns the name of the directory containing the file
    return os.path.basename(os.path.dirname(path))


## Analysis Operations
def count_labels(corpus):
    labels = corpus.map(itemgetter(0)).map(parse_label)
    label_count = labels.map(lambda l: (l, 1)).reduceByKey(add)

    table = [["Label", "Count"]]
    table.extend([
        [label, count] for label, count in label_count.collect()
    ])
    print(tabulate(table))


def count_bigrams(corpus):
    text = corpus.map(itemgetter(1))
    sents = text.flatMap(nltk.sent_tokenize)
    sents = sents.map(lambda s: list(nltk.word_tokenize(s)))

    bigrams = sents.flatMap(lambda s: list(nltk.bigrams(s)))
    unique_bigrams = bigrams.distinct().count()
    print("unique bigrams: {}".format(unique_bigrams))

    bigram_counts = bigrams.map(lambda g: (g, 1)).reduceByKey(add).toDF()
    print(bigram_counts.head())


## Main functionality
def main(sc, spark):
    corpus = sc.wholeTextFiles(CORPUS)

    # Count the labels
    count_labels(corpus)

    # Perform the bigram count
    count_bigrams(corpus)



if __name__ == "__main__":
    # Configure Spark
    conf  = SparkConf().setAppName(APP_NAME)
    conf  = conf.setMaster("local[*]")
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Execute Main functionality
    main(sc, spark)
