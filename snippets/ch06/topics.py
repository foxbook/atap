#!/usr/bin/env python3

from reader import PickledCorpusReader
from transformers import TextNormalizer, GensimTfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer

from gensim.sklearn_api import lsimodel, ldamodel

def identity(words):
    return words


class SklearnTopicModels(object):

    def __init__(self, n_topics=50, estimator='LDA'):
        """
        n_topics is the desired number of topics
        To use Latent Semantic Analysis, set estimator to 'LSA',
        To use Non-Negative Matrix Factorization, set estimator to 'NMF',
        otherwise, defaults to Latent Dirichlet Allocation ('LDA').
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_topics=self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('tfidf', CountVectorizer(tokenizer=identity,
                                      preprocessor=None, lowercase=False)),
            ('model', self.estimator)
        ])


    def fit_transform(self, documents):
        self.model.fit_transform(documents)

        return self.model


    def get_topics(self, n=25):
        """
        n is the number of top terms to show for each topic
        """
        vectorizer = self.model.named_steps['tfidf']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics


class GensimTopicModels(object):

    def __init__(self, n_topics=50, estimator='LDA'):
        """
        n_topics is the desired number of topics

        To use Latent Semantic Analysis, set estimator to 'LSA'
        otherwise defaults to Latent Dirichlet Allocation.
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = lsimodel.LsiTransformer(num_topics=self.n_topics)
        else:
            self.estimator = ldamodel.LdaTransformer(num_topics=self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', GensimTfidfVectorizer()),
            ('model', self.estimator)
        ])

    def fit(self, documents):
        self.model.fit(documents)

        return self.model


if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')

    # With Sklearn
    skmodel = SklearnTopicModels(estimator='NMF')
    documents   = corpus.docs()

    skmodel.fit_transform(documents)
    topics = skmodel.get_topics()
    for topic, terms in topics.items():
        print("Topic #{}:".format(topic+1))
        print(terms)

    # # With Gensim
    # gmodel = GensimTopicModels(estimator='LSA')
    #
    # docs = [
    #     list(corpus.docs(fileids=fileid))[0]
    #     for fileid in corpus.fileids()
    # ]
    #
    # gmodel.fit(docs)
    #
    # # retrieve the fitted lsa model from the named steps of the pipeline
    # lsa = gmodel.model.named_steps['lsa'].gensim_model
    #
    # # show the topics with the token-weights for the top 10 most influential tokens:
    # print(lsa.print_topics(10))


    # # retrieve the fitted lda model from the named steps of the pipeline
    # lda = gmodel.model.named_steps['lda'].gensim_model
    #
    # # show the topics with the token-weights for the top 10 most influential tokens:
    # lda.print_topics(10)

    # corpus = [
    #     gmodel.model.named_steps['vect'].lexicon.doc2bow(doc)
    #     for doc in gmodel.model.named_steps['norm'].transform(docs)
    # ]
    #
    #
    # id2token = gmodel.model.named_steps['vect'].lexicon.id2token
    #
    # for word_id, freq in next(iter(corpus)):
    #     print(id2token[word_id], freq)

    # # get the highest weighted topic for each of the documents in the corpus
    # def get_topics(vectorized_corpus, model):
    #     from operator import itemgetter
    #
    #     topics = [
    #         max(model[doc], key=itemgetter(1))[0]
    #         for doc in vectorized_corpus
    #     ]
    #
    #     return topics
    #
    # topics = get_topics(corpus,lda)
    #
    # for topic, doc in zip(topics, docs):
    #     print("Topic:{}".format(topic))
    #     print(doc)
    #
    ## retreive the fitted vectorizer or the lexicon if needed
    # tfidf = gmodel.model.named_steps['vect'].tfidf
    # lexicon = gmodel.model.named_steps['vect'].lexicon
