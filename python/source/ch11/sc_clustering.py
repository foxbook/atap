## Spark Application - execute with spark-submit

## Imports
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import Word2Vec, Tokenizer

from sc_vectorization import load_corpus

from tabulate import tabulate
from functools import partial
tabulate = partial(tabulate, tablefmt="simple", headers="firstrow")

## Module Constants
APP_NAME = "Text Clustering"
CORPUS = "text_corpus/corpus/*/*.txt"


## Main functionality
def main(sc, spark):
    # Load the Corpus
    corpus = load_corpus(sc, spark)

    # Create the vector/cluster pipeline
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="tokens"),
        Word2Vec(vectorSize=7, minCount=0, inputCol="tokens", outputCol="vecs"),
        BisectingKMeans(k=10, featuresCol="vecs", maxIter=10),
    ])

    # Fit the model
    model = pipeline.fit(corpus)
    corpus = model.transform(corpus)

    # Evaluate clustering.
    bkm = model.stages[-1]
    cost = bkm.computeCost(corpus)
    sizes = bkm.summary.clusterSizes

    # TODO: compute cost of each cluster individually

    # Get the text representation of each cluster.
    wvec = model.stages[-2]
    table = [["Cluster", "Size", "Terms"]]
    for ci, c in enumerate(bkm.clusterCenters()):
        ct = wvec.findSynonyms(c, 7)
        size = sizes[ci]
        terms = " ".join([row.word for row in ct.take(7)])
        table.append([ci, size, terms])

    # Print Results
    print(tabulate(table))
    print("Sum of square distance to center: {:0.3f}".format(cost))


if __name__ == "__main__":
    # Configure Spark
    conf  = SparkConf().setAppName(APP_NAME)
    conf  = conf.setMaster("local[*]")
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Execute Main functionality
    main(sc, spark)
