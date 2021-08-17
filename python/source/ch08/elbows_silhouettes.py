import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.base import Bunch
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

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

corpus = load_corpus('hobbies')
tfidf  = TfidfVectorizer(stop_words='english')
docs   = tfidf.fit_transform(corpus.data)

# Instantiate the clustering model and visualizer
visualizer = SilhouetteVisualizer(KMeans(n_clusters=6))
visualizer.fit(docs)
visualizer.poof()

# Instantiate the clustering model and visualizer
visualizer = KElbowVisualizer(KMeans(), metric='silhouette', k=[4,10])
visualizer.fit(docs)
visualizer.poof()
