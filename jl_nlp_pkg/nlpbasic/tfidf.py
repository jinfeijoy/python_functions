import gensim
from gensim.models import TfidfModel
import pandas as pd


def get_tfidf_dataframe(corpus, no_below =5, doc_index = None,
                        no_above = 0.5, keep_n = 100000):
    """
    :param corpus: corpus: a list of list of tokens (generated from textClean.pipeline)
    :param no_below: filter out tokens that less than no_below documents (absolute number)
    :param doc_index: document index
    :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
    :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    :return: tfidf dataframe with document id, bag of words and tfidf value
    """

    processed_docs = corpus
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below,
                               no_above=no_above,
                               keep_n=keep_n)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = TfidfModel(bow_corpus)
    vector = tfidf[bow_corpus]
    doc_id = []
    bow = []
    tfidf_value = []
    for index, doc in enumerate(vector):
        for id, value in doc:
            if doc_index == None:
                doc_id.append(index)
            else:
                doc_id.append(doc_index[index])
            bow.append(dictionary.get(id))
            tfidf_value.append(value)
    data = {"doc_id": doc_id, "bow": bow, "tfidf_value": tfidf_value}
    data = pd.DataFrame(data)
    return data

def get_top_n_tfidf_bow(corpus, no_below = 5,
                        no_above = 0.5, keep_n = 100000, top_n_tokens = 100000):
    """
    :param corpus: corpus: a list of list of tokens (generated from textClean.pipeline)
    :param no_below: filter out tokens that less than no_below documents (absolute number)
    :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
    :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    :param top_n_tokens: top n bag of words in tfidf global list
    :return: a list of tokens with global top n tfidf value
    """
    tmp_data = get_tfidf_dataframe(corpus, no_below = no_below, no_above = no_above, keep_n = keep_n)
    tfidf_max_value = tmp_data.groupby('bow')['tfidf_value'].nlargest(1).reset_index(drop=False).sort_values(
        by='tfidf_value', ascending=False)
    output_list = tfidf_max_value.head(top_n_tokens).bow.tolist()
    return output_list

