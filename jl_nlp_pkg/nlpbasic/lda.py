import gensim
from gensim.models import TfidfModel
from nltk.stem.porter import *
import pandas as pd
from nlpbasic import tfidf



def fit_lda(corpus, no_below = 5, no_above = 0.5, keep_n = 100000, top_n_tokens = '', num_topics = 5):
    """
    :param corpus: corpus: a list of list of tokens (generated from textClean.pipeline)
    :param no_below: filter out tokens that less than no_below documents (absolute number)
    :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
    :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    :param top_n_tokens: top n bag of words in tfidf global list
    :param num_topics: number of topics in lda model
    :return: lda model file, bow_corpus, dictionary
    """
    if top_n_tokens == '':
        processed_docs = corpus
    else:
        selected_tokens = tfidf.get_top_n_tfidf_bow(corpus, no_below, no_above, keep_n, top_n_tokens)
        processed_docs = [list(i for i in token if i in selected_tokens) for token in corpus]
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above,
                               keep_n=keep_n)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           num_topics=num_topics,
                                           id2word=dictionary,
                                           passes=2,
                                           workers=2,
                                           random_state=100)
    return lda_model, bow_corpus, dictionary


def lda_topics(lda_model):
    """
    :return: lda model topics with top 10 words
    """
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topics.append(topic)
    toptopic = pd.DataFrame({'topics': topics})  #
    toptopic[['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9',
              'var10']] = toptopic.topics.str.split("+", expand=True)
    toptopic = toptopic.drop(columns=['topics'])
    toptopic = toptopic.applymap(lambda x: re.search('"(.*)"', x).group(1))
    toptopic.insert(0, 'Topics', toptopic.index + 1)
    return toptopic


