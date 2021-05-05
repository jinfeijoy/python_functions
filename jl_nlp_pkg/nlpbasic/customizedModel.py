import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pandas as pd
from functools import partial
from nlpbasic.TextProcessing import TextProcessing
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cosine

class customizedTFIDF(object):

    def __init__(self):
        pass

    def get_tfidf_dataframe(data, multi_gram, stem_lemma = '', tag_drop =[],
                            ngram_tag_drop = False, no_below =5,
                            no_above = 0.5, keep_n = 100000):
        """
        :param data: the document column in dataset to be calculate tfidf, eg data['doc']
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param no_below: filter out tokens that less than no_below documents (absolute number)
        :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
        :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
        :return: tfidf dataframe with document id, bag of words and tfidf value
        """

        processed_docs = data.map(
            partial(TextProcessing.find_multiple_gram_tokens,
                    multi_gram=multi_gram,
                    stem_lemma=stem_lemma,
                    tag_drop=tag_drop,
                    ngram_tag_drop = ngram_tag_drop
                    )
        )
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
                doc_id.append(index)
                bow.append(dictionary.get(id))
                tfidf_value.append(value)
        data = {"doc_id": doc_id, "bow": bow, "tfidf_value": tfidf_value}
        data = pd.DataFrame(data)
        return (data)

    def get_top_n_tfidf_bow(data, multi_gram, stem_lemma = '', tag_drop =[],
                            ngram_tag_drop = False, no_below = 5,
                            no_above = 0.5, keep_n = 100000, top_n_tokens = 100000):
        """
        :param data: the document column in dataset to be calculate tfidf, eg data['doc']
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param no_below: filter out tokens that less than no_below documents (absolute number)
        :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
        :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
        :param top_n_tokens: top n bag of words in tfidf global list
        :return: a list of tokens with global top n tfidf value
        """
        tmp_data = customizedTFIDF.get_tfidf_dataframe(data, multi_gram, stem_lemma , tag_drop ,
                                                       ngram_tag_drop , no_below , no_above , keep_n)
        tfidf_max_value = tmp_data.groupby('bow')['tfidf_value'].nlargest(1).reset_index(drop=False).sort_values(
            by='tfidf_value', ascending=False)
        output_list = tfidf_max_value.head(top_n_tokens).bow.tolist()
        return output_list

    def get_similarity_cosin(basedata, filterdata, key, doc_key, comp_col, topn_topics=10):
        """
        :param basedata: basedata to do comparison
        :param filterdata: compare data to do comparison
        :param key: key used to do left join e.g. ['A'], ['A','B']
        :param doc_key: index key which used to identify rows
        :param comp_col: comparison column e.g.'A'
        :param topn_topics: rows with top n similarity value
        :return: a dataset with row doc key and similarity value
        """
        basedata['comp_col'] = basedata[comp_col]
        filterdata['comp_col'] = filterdata[comp_col]
        merge_data = pd.merge(basedata, filterdata, how='left', left_on=key, right_on=key, suffixes=('_x', '_y'))
        merge_data.fillna(0, inplace=True)
        similarity_val = merge_data.groupby([doc_key]).apply(lambda x: 1 - cosine(x['comp_col_x'], x['comp_col_y']))
        similarity_data = {"doc_key": similarity_val.index, "cosine": similarity_val}
        similarity_data = pd.DataFrame(similarity_data)
        similarity_data = similarity_data.sort_values(by=['cosine'], ascending=False).head(topn_topics)
        return similarity_data





class customizedLDA(object):

    def __init__(self):
        pass

    def print_bow_example(text, dictionary):
        """
        :param dictionary: the dictionary generated by gensim.corpora.Dictionary(processed_docs)
        :return: print bag of words vector frequency
        """
        for i in range(len(text)):
            print("Word {} (\"{}\") appears {} time.".format(text[i][0], dictionary[text[i][0]], text[i][1]))

    def fit_lda(data, multi_gram, stem_lemma = '', tag_drop =[],
                ngram_tag_drop = False, no_below = 5,
                no_above = 0.5, keep_n = 100000, top_n_tokens = 100000, num_topics = 5):
        """
        :param data: the document column in dataset to be calculate tfidf, eg data['doc']
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param no_below: filter out tokens that less than no_below documents (absolute number)
        :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
        :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
        :param top_n_tokens: top n bag of words in tfidf global list
        :param num_topics: number of topics in lda model
        :return: lda model file, bow_corpus, dictionary
        """
        if top_n_tokens == '':
            processed_docs = data.map(
                partial(TextProcessing.find_multiple_gram_tokens,
                        multi_gram=multi_gram,
                        stem_lemma=stem_lemma,
                        tag_drop=tag_drop,
                        ngram_tag_drop = ngram_tag_drop))
        else:
            selected_tokens = customizedTFIDF.get_top_n_tfidf_bow(data, multi_gram, stem_lemma, tag_drop,
                                                  ngram_tag_drop, no_below, no_above, keep_n, top_n_tokens)
            processed_docs = data.map(partial(TextProcessing.keep_specific_tokens,
                                              multi_gram=multi_gram,
                                              stem_lemma=stem_lemma,
                                              tag_drop=tag_drop,
                                              ngram_tag_drop=ngram_tag_drop,
                                              selected_tokens=selected_tokens))
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
        # for idx, topic in lda_model.print_topics(-1):
        #     print('Topic: {} \nWords: {}'.format(idx, topic))
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
