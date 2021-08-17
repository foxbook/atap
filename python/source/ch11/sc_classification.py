## Spark Application - execute with spark-submit

## Imports
from tabulate import tabulate
from functools import partial
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from sc_vectorization import load_corpus, make_vectorizer

tabulate = partial(tabulate, tablefmt="simple", headers="firstrow")

## Module Constants
APP_NAME = "Text Classification"
CORPUS = "text_corpus/corpus/*/*.txt"

## Main functionality
def main(sc, spark):
    # Load and vectorize the corpus
    corpus = load_corpus(sc, spark)
    vector = make_vectorizer().fit(corpus)

    # Index the labels of the classification
    labelIndex = StringIndexer(inputCol="label", outputCol="indexedLabel")
    labelIndex = labelIndex.fit(corpus)

    # Split the data into training and test sets
    training, test = corpus.randomSplit([0.8, 0.2])

    # Create the classifier
    clf = LogisticRegression(
        maxIter=10, regParam=0.3, elasticNetParam=0.8,
        family="multinomial", labelCol="indexedLabel", featuresCol="tfidf")

    # Create the model
    model = Pipeline(stages=[
        vector, labelIndex, clf
    ]).fit(training)

    # Make predictions
    predictions = model.transform(test)
    predictions.select("prediction", "indexedLabel", "tfidf").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    gbtModel = model.stages[2]
    print(gbtModel)  # summary only


if __name__ == "__main__":
    # Configure Spark
    conf  = SparkConf().setAppName(APP_NAME)
    conf  = conf.setMaster("local[*]")
    sc    = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Execute Main functionality
    main(sc, spark)
