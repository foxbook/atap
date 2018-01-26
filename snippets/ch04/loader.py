from sklearn.cross_validation import KFold

class CorpusLoader(object):

    def __init__(self, corpus, folds=None, shuffle=True):
        self.n_docs = len(corpus.fileids())
        self.corpus = corpus
        self.folds  = folds

        if folds is not None:
            # Generate the KFold cross validation for the loader.
            self.folds = KFold(self.n_docs, folds, shuffle)

    @property
    def n_folds(self):
        """
        Returns the number of folds if it exists; 0 otherwise.
        """
        if self.folds is None: return 0
        return self.folds.n_folds

    def fileids(self, fold=None, train=False, test=False):

        if fold is None:
            # If no fold is specified, return all the fileids.
            return self.corpus.fileids()

        # Otherwise, identify the fold specifically and get the train/test idx
        train_idx, test_idx = [split for split in self.folds][fold]

        # Now determine if we're in train or test mode.
        if not (test or train) or (test and train):
            raise ValueError(
                "Please specify either train or test flag"
            )

        # Select only the indices to filter upon.
        indices = train_idx if train else test_idx
        return [
            fileid for doc_idx, fileid in enumerate(self.corpus.fileids())
            if doc_idx in indices
        ]

    def documents(self, fold=None, train=False, test=False):
        for fileid in self.fileids(fold, train, test):
            yield list(self.corpus.docs(fileids=fileid))

    def labels(self, fold=None, train=False, test=False):
        return [
            self.corpus.categories(fileids=fileid)[0]
            for fileid in self.fileids(fold, train, test)
        ]


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('corpus')
    loader = CorpusLoader(corpus, 12)

    for fid in loader.fileids(0, test=True):
        print(fid)
