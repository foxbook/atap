## Spark Application - execute with spark-submit

## Imports
import os

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, NGram
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

## Module Constants
APP_NAME = "Text Vectorization"
CORPUS = "text_corpus/corpus/*/*.txt"

## Closure Functions
def parse_label(path):
    # Returns the name of the directory containing the file
    return os.path.basename(os.path.dirname(path))


## Data Manipulation
def load_corpus(sc, spark, path=CORPUS):
    # Load data from disk and transform into a DataFrame
    data = sc.wholeTextFiles(path)
    corpus = data.map(lambda d: (parse_label(d[0]), d[1]))
    return spark.createDataFrame(corpus, ["label", "text"])


def make_vectorizer(stopwords=True, tfidf=True, n_features=5000):
    # Creates a vectorization pipeline that starts with tokenization
    stages = [
        Tokenizer(inputCol="text", outputCol="tokens"),
    ]

    # Append stopwords to the pipeline if requested
    if stopwords:
        stages.append(
            StopWordsRemover(
                caseSensitive=False, outputCol="filtered_tokens",
                inputCol=stages[-1].getOutputCol(),
            ),
        )

    # Create the Hashing term frequency vectorizer
    stages.append(
        HashingTF(
            numFeatures=n_features,
            inputCol=stages[-1].getOutputCol(),
            outputCol="frequency"
        )
    )

    # Append the IDF vectorizer if requested
    if tfidf:
        stages.append(
            IDF(inputCol=stages[-1].getOutputCol(), outputCol="tfidf")
        )

    # Return the completed pipeline
    return Pipeline(stages=stages)


## Main functionality
def main(sc, spark):
    corpus = load_corpus(sc, spark)
    vector = vectorize().fit(corpus)

    corpus = vector.transform(corpus)
    print(corpus.head())



if __name__ == "__main__":
    # Configure Spark
    conf  = SparkConf().setAppName(APP_NAME)
    conf  = conf.setMaster("local[*]")
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Execute Main functionality
    main(sc, spark)
