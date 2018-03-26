import os
from sklearn.datasets.base import Bunch
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

# The path to the test data sets
FIXTURES = os.path.join(os.getcwd(), "data")

# Corpus loading mechanisms
corpora = {
    "hobbies": os.path.join(FIXTURES, "hobbies")
}


def load_corpus(name, download=True):
    """
    Loads and wrangles the passed in text corpus by name.
    If download is specified, this method will download any missing files.

    Note: This function is slightly different to the `load_data` function
    used above to load pandas dataframes into memory.
    """

    # Get the path from the datasets
    path = corpora[name]

    # Check if the data exists, otherwise download or raise
    if not os.path.exists(path):
        raise ValueError((
            "'{}' dataset has not been downloaded, "
            "use the download.py module to fetch datasets"
        ).format(name))

    # Read the directories in the directory as the categories.
    categories = [
        cat for cat in os.listdir(path)
        if os.path.isdir(os.path.join(path, cat))
    ]

    files = []  # holds the file names relative to the root
    data = []  # holds the text read from the file
    target = []  # holds the string of the category

    # Load the data from the files in the corpus
    for cat in categories:
        for name in os.listdir(os.path.join(path, cat)):
            files.append(os.path.join(path, cat, name))
            target.append(cat)

            with open(os.path.join(path, cat, name), 'r') as f:
                data.append(f.read())

    # Return the data bunch for use similar to the newsgroups example
    return Bunch(
        categories=categories,
        files=files,
        data=data,
        target=target,
    )


# Load the data and create document vectors
corpus = load_corpus('hobbies')
tfidf  = TfidfVectorizer()

docs   = tfidf.fit_transform(corpus.data)
labels = corpus.target

# Create a visualizer to simply see the vectors plotted in 2D
tsne = TSNEVisualizer()
tsne.fit(docs)
tsne.poof()


# Create a visualizer to see how k-means clustering grouped the docs
from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=5)
clusters.fit(docs)

tsne = TSNEVisualizer()
tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
tsne.poof()


# Create a visualizer to see how the classes are distributed
tsne = TSNEVisualizer()
tsne.fit(docs, labels)
tsne.poof()
