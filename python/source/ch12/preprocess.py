import os
import nltk
import pickle
import multiprocessing as mp

from slugify import slugify


class Preprocessor(object):
    """
    The preprocessor wraps a SqliteCorpusReader and manages the stateful
    tokenization and part of speech tagging into a directory that is stored
    in a format that can be read by the `PickledCorpusReader`. This format
    is more compact and necessarily removes a variety of fields from the
    document that are stored in Sqlite database. This format however is more
    easily accessed for common parsing activity.
    """

    def __init__(self, corpus, target=None, **kwargs):
        """
        The corpus is the `SqliteCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.tasks = mp.cpu_count()
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

    def abspath(self, name):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        # Create the pickle file extension
        fname  = str(name) + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, fname))

    def tokenize(self, text):
        """
        Segments, tokenizes, and tags a document in the corpus. Returns a
        generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        yield [
            nltk.pos_tag(nltk.wordpunct_tokenize(sent))
            for sent in nltk.sent_tokenize(text)
        ]

    def process(self, score_album_artist_text):
        """
        For a single file does the following preprocessing work:
            1. Checks the location on disk to make sure no errors occur.
            2. Gets all paragraphs for the given text.
            3. Segments the paragraphs with the sent_tokenizer
            4. Tokenizes the sentences with the wordpunct_tokenizer
            5. Tags the sentences using the default pos_tagger
            6. Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        score, album, artist, text = score_album_artist_text

        # Compute the outpath to write the file to.
        if album:
            name = album+'-'+artist
        else:
            name = artist
        target = self.abspath(slugify(name))
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
        document = list(self.tokenize(text))
        document.append(score)

        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Clean up the document
        del document

        # Return the target fileid
        return target

    def transform(self):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        This method will also directly copy files that are in the corpus.root
        directory that are not matched by the corpus.fileids().
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        for score_album_artist_text in self.corpus.scores_albums_artists_texts():
            yield self.process(score_album_artist_text)


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

    def transform(self):
        """
        Create a pool using the multiprocessing library, passing in
        the number of cores available to set the desired number of
        processes.
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Reset the results
        self.results = []

        # Create a multiprocessing pool
        pool  = mp.Pool(processes=self.tasks)
        tasks = [
            pool.apply_async(self.process, (score_album_artist_text,), callback=self.on_result)
            for score_album_artist_text in self.corpus.scores_albums_artists_texts()
        ]

        # Close the pool and join
        pool.close()
        pool.join()

        return self.results


if __name__ == '__main__':

    from reader import SqliteCorpusReader

    corpus = SqliteCorpusReader('../database.sqlite')
    transformer = Preprocessor(corpus, '../review_corpus_proc')
    docs = transformer.transform()
    print(len(list(docs)))