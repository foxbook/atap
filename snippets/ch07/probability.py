import math
import numpy as np

from tabulate import tabulate
from collections import Counter


_NINF = float('-1e300')

##########################################################################
## FreqDist Class
##########################################################################

class FreqDist(Counter):

    def incr(self, item, amount=1):
        """
        Increment the count of the specified item by the specified amount.
        """
        self[item] += amount

    def decr(self, item, amount=1):
        """
        Decrement the count of the specified item by the specified amount.
        """
        self[item] -= amount

    def N(self):
        """
        Return the total number of sample outcomes recorded.
        """
        return sum(self.values())

    def B(self):
        """
        Return the total number of sample values (or "bins") that have
        associated counts that are greater than zero.
        """
        return len(self)

    def max(self):
        """
        Return the sample with the largest count. If two samples have the
        same count then only one is returned; which one is undefined.
        """
        if len(self) == 0:
            raise ValueError(
                "must have at least one item before max is defined."
            )
        return self.most_common(1)[0][0]

    def hapaxes(self):
        """
        Return a list of samples that occur once (hapax legomena)
        """
        return [item for item in self if self[item] == 1]

    def freq(self, sample):
        """
        Returns the relative frequency (also the empirical probability) of
        the given sample, that is the count of the item divided by the total
        number of outcomes.
        """
        n = self.N()
        if n == 0:
            return 0.0
        return self[sample] / n

    def cumfreq(self, samples=None):
        """
        Returns the cumulative absolute frequencies of of the samples
        specified. If samples is None, returns all samples in order of
        frequency.
        """
        # Collect all samples if non specified
        samples = samples or [elem[0] for elem in self.most_common()]

        # Compute and yield the absolute cumulative frequency
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    def logprob(self, sample):
        """
        Return the base 2 log of the probability given by the maximum
        likelihood estimate. The log probability is used to improve
        computational performance.
        """
        p = self.freq(sample)
        return (math.log(p, 2) if p != 0 else _NINF)

    def tabulate(self, samples=None, cumulative=False):
        """
        Return a table of the frequency distribution, with most frequent
        samples first. If samples is a list, then filter the list to those
        samples, if it is an int, limit the list to that number of samples.
        If cumfreq is true, use the cumulative frequency rather than the
        relative frequency.
        """
        if samples is None or isinstance(samples, int):
            # Get the N most common smaples
            samples = [elem[0] for elem in self.most_common(samples)]
        else:
            # Sort the samples by frequency descending
            samples = sorted(samples, key=lambda s: self[s], reverse=True)

        table = [["Sample", "Frequency"]]
        if cumulative:
            freqs = list(self.cumfreq(samples))
        else:
            freqs = [self.freq(sample) for sample in samples]

        table += list(zip(samples, freqs))
        return tabulate(table, headers='firstrow')

    def plot(self, samples=None, cumulative=False, title=None):
        """
        Plot the samples from the frequency distribution.
        """

        # Load matplotlib, exit if there is an import error.
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter, MaxNLocator
        except ImportError:
            raise ValueError(
                "the plot function requires matplotlib to be installed"
            )

        # Get desired samples from distribution
        if samples is None or isinstance(samples, int):
            # Get the N most common smaples
            samples = [elem[0] for elem in self.most_common(samples)]
        else:
            # Sort the samples by frequency descending
            samples = sorted(samples, key=lambda s: self[s], reverse=True)

        # Compute the frequency values
        if cumulative:
            freqs = list(self.cumfreq(samples))
            ylabel = "Cumulative Counts"
            title = title or "Cumulative Frequency Distribution"
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = "Counts"
            title = title or "Frequency Distribution"

        # Closure for assigning tick labels
        def tick_fmt(tick_val, tick_pos):
            if int(tick_val) < len(samples):
                return samples[int(tick_val)]
            return ''

        # Build the plot
        fig, ax = plt.subplots(figsize=(9,6))
        ax.plot(freqs, lw=2, c="b", ls="-", label="frequency")

        # Set the x-axis labels with the formatter
        n = len(samples)
        ax.xaxis.set_major_formatter(FuncFormatter(tick_fmt))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
        fig.subplots_adjust(bottom=0.18)

        # Set axis labels and title and legend
        ax.set_xlabel("Samples")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        return ax


##########################################################################
## Examples from Text
##########################################################################

def describe(corpus):
    """
    Returns the word count, size of the vocabulary, and lexical diversity.
    """
    counts = FreqDist()
    for word in corpus.words():
        counts.incr(word)

    print("max is: {}".format(counts.max()))
    print("top 10:")
    print(counts.most_common(10))
    print("number of hapaxes: {}".format(len(counts.hapaxes())))
    print("first ten hapaxes:")
    print(counts.hapaxes()[:10])

    return {
        'words': counts.N(),
        'vocab': counts.B(),
        'lexdiv': counts.N() / counts.B(),
    }


def freqtable(corpus):
    counts = FreqDist([word.lower() for word in corpus.words()])
    terms = ["iphone", "android", "apple", "microsoft", "amazon", "google"]
    print(counts.tabulate(samples=terms))


def freqplot(corpus):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    counts = FreqDist([word.lower() for word in corpus.words()])
    samples = [
        elem[0] for elem in counts.most_common()
    ][50:]

    g = counts.plot(samples=samples, cumulative=True)
    plt.savefig("cumfreq.png")

    g = counts.plot(samples=samples, cumulative=False)
    plt.savefig("freq.png")


if __name__ == '__main__':
    from reader import PickledCorpusReader
    corpus = PickledCorpusReader('../corpus')

    # Describe example
    # print(describe(corpus))

    # Frequency Table
    freqtable(corpus)

    # Plotting Example
    # freqplot(corpus)
