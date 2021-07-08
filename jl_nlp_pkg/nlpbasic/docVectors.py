import gensim
import pandas as pd
import numpy as np

def get_vocab(tokens):
    '''
    this function is to get
    :param tokens: a list of list of tokens (generated from textClean.pipeline)
    :return: list of vocabulary
    '''
    vocab = []
    total_words = 0
    for token in tokens:
        total_words = total_words + len(token)
        for i in range(len(token)):
            if token[i] not in vocab:
                vocab.append(token[i])
    return vocab

def generate_corpus_dict(corpus, no_below =5,
                        no_above = 0.5, keep_n = 100000):
    """
    :param corpus: a list of list of tokens (generated from textClean.pipeline)
    :param no_below: filter out tokens that less than no_below documents (absolute number)
    :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
    :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    :return: bow corpus, dictionary
    """
    dictionary = gensim.corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=no_below,
                               no_above=no_above,
                               keep_n=keep_n)
    return dictionary

def create_document_vector(corpus, dictionary):
    """
    :param corpus: a list of list of tokens (generated from textClean.pipeline)
    :param dictionary: dictionary generated from generate_corpus_dict()
    :return:
    """
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    return bow_corpus

def create_corpus_vector(corpus, dictionary):
    """
    :param corpus: a list of list of tokens (generated from textClean.pipeline)
    :param dictionary:
    :return:
    """
    return [create_document_vector(doc, dictionary) for doc in corpus]


def get_vocab_matrix(document_vector, vocabulary):
    """
    :param document_vector: bow_corpus generated from create_document_vector
    :param vocabulary: dictionary generated from generate_corpus_dict
    :return: term document matrix
    """
    my_matrix = pd.DataFrame(0.0, index=np.arange(len(document_vector)), columns=[i for i in vocabulary.values()])
    for i in range(len(document_vector)):
        for j in range(len(document_vector[i])):
            my_matrix.at[i, vocabulary[document_vector[i][j][0]]] = document_vector[i][j][1]
    my_matrix.index.name = 'doc'
    return my_matrix