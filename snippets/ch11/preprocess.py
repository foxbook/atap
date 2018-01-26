import os
import nltk
import shutil
import pickle
import multiprocessing as mp

from tqdm import tqdm


class Preprocessor(object):
    """
    The preprocessor wraps a corpus object (usually a `HTMLCorpusReader`)
    and manages the stateful tokenization and part of speech tagging into a
    directory that is stored in a format that can be read by the
    `HTMLPickledCorpusReader`. This format is more compact and necessarily
    removes a variety of fields from the document that are stored in the JSON
    representation dumped from the Mongo database. This format however is more
    easily accessed for common parsing activity.
    """

    def __init__(self, corpus, target=None, **kwargs):
        """
        The corpus is the `HTMLCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, path):
        if path is not None:
            # Normalize the path and make it absolute
            path = os.path.expanduser(path)
            path = os.path.expandvars(path)
            path = os.path.abspath(path)

            if os.path.exists(path):
                if not os.path.isdir(path):
                    raise ValueError(
                        "Please supply a directory to write preprocessed data to."
                    )

        self._target = path

    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        # Find the directory, relative from the corpus root.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        # Compute the name parts to reconstruct
        basename  = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Create the pickle file extension
        basename  = name + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def replicate(self, source):
        """
        Directly copies all files in the source directory to the root of the
        target directory (does not maintain subdirectory structures). Used to
        copy over metadata files from the root of the corpus to the target.
        """
        names = [
            name for name in os.listdir(source)
            if not name.startswith('.')
        ]

        # Filter out directories and copy files
        for name in names:
            src = os.path.abspath(os.path.join(source, name))
            dst = os.path.abspath(os.path.join(self.target, name))

            if os.path.isfile(src):
                shutil.copy(src, dst)

    def tokenize(self, fileid):
        """
        Segments, tokenizes, and tags a document in the corpus. Returns a
        generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                nltk.pos_tag(nltk.wordpunct_tokenize(sent))
                for sent in nltk.sent_tokenize(paragraph)
            ]

    def process(self, fileid):
        """
        For a single file does the following preprocessing work:
            1. Checks the location on disk to make sure no errors occur.
            2. Gets all paragraphs for the given text.
            3. Segements the paragraphs with the sent_tokenizer
            4. Tokenizes the sentences with the wordpunct_tokenizer
            5. Tags the sentences using the default pos_tagger
            6. Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Make sure the directory exists
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Make sure that the parent is a directory and not a file
        if not os.path.isdir(parent):
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )


        # Create a data structure for the pickle
        document = list(self.tokenize(fileid))

        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Clean up the document
        del document

        # Return the target fileid
        return target

    def transform(self, fileids=None, categories=None):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        This method will also directly copy files that are in the corpus.root
        directory that are not matched by the corpus.fileids().
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # First shutil.copy anything in the root directory.
        self.replicate(self.corpus.root)

        # Resolve the fileids to start processing
        for fileid in self.fileids(fileids, categories):
            yield self.process(fileid)


class ProgressPreprocessor(Preprocessor):
    """
    This preprocessor adds a progress bar for visually informing the user
    what is going on during preprocessing.
    """

    def transform(self, fileids=None, categories=None):
        """
        At the moment, we simply have to replace the entire transform method
        to get progress bar functionality. Kind of a bummer, but it's a small
        method (purposefully so).
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # First shutil.copy anything in the root directory.
        self.replicate(self.corpus.root)

        # Get the total corpus size for per byte counting
        corpus_size = sum(self.corpus.sizes(fileids, categories))

        # Start processing with a progress bar.
        with tqdm(total=corpus_size, unit='B', unit_scale=True) as pbar:
            for fileid in self.fileids(fileids, categories):
                yield self.process(fileid)
                pbar.update(sum(self.corpus.sizes(fileids=fileid)))


class ParallelPreprocessor(Preprocessor):
    """
    Implements multiprocessing to speed up the preprocessing efforts.
    """

    def __init__(self, *args, **kwargs):
        """
        Get parallel-specific arguments and then call super.
        """
        self.tasks = mp.cpu_count()
        super(ParallelPreprocessor, self).__init__(*args, **kwargs)

    def on_result(self, result):
        """
        Appends the results to the master results list.
        """
        self.results.append(result)

    def transform(self, fileids=None, categories=None):
        """
        Create a pool using the multiprocessing library, passing in
        the number of cores available to set the desired number of
        processes.
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # First shutil.copy anything in the root directory.
        self.replicate(self.corpus.root)

        # Reset the results
        self.results = []

        # Create a multiprocessing pool
        pool  = mp.Pool(processes=self.tasks)
        tasks = [
            pool.apply_async(self.process, (fileid,), callback=self.on_result)
            for fileid in self.fileids(fileids, categories)
        ]

        # Close the pool and join
        pool.close()
        pool.join()

        return self.results


class ProgressParallelPreprocessor(ParallelPreprocessor):
    """
    Preprocessor that implements both multiprocessing and a progress bar.
    Note: had to jump through a lot of hoops just to get a progress bar, not
    sure it was worth it or that this performs the most effectively ...
    """

    def on_result(self, pbar):
        """
        Indicates progress on result.
        """

        def inner(result):
            pbar.update(1)
            self.results.append(result)
        return inner

    def transform(self, fileids=None, categories=None):
        """
        Setup the progress bar before conducting multiprocess transform.
        """

        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # First shutil.copy anything in the root directory.
        self.replicate(self.corpus.root)

        # Reset the results
        self.results = []
        fileids = self.fileids(fileids, categories)

        # Get the total corpus size for per byte counting and create pbar
        with tqdm(total=len(fileids), unit='Docs') as pbar:

            # Create a multiprocessing pool
            pool  = mp.Pool(processes=self.tasks)
            tasks = [
                pool.apply_async(self.process, (fileid,), callback=self.on_result(pbar))
                for fileid in fileids
            ]

            # Close the pool and join
            pool.close()
            pool.join()

        return self.results


if __name__ == '__main__':

    from reader import HTMLCorpusReader

    corpus = HTMLCorpusReader('../food_corpus')
    transformer = ProgressParallelPreprocessor(corpus, '../food_corpus_proc')
    docs = transformer.transform()
    print(len(list(docs)))