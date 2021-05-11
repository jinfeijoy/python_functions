import numpy as np
import statistics
from nlpbasic.TextProcessing import TextProcessing
from functools import partial
import nltk
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

class DataExploration(object):
    def __init__(self):
        pass

    def text_length_summary(data, col, measure = 'max'):
        """

        :param col: column of data that want to be measured: e.g. "review"
        :param measure: 'max', 'min', 'avg', 'median'
        :return:
        """
        measurer = np.vectorize(len)
        if measure == 'max':
            out = measurer(data[col].astype(str)).max(axis=0)
        if measure == 'min':
            out = measurer(data[col].astype(str)).min(axis=0)
        if measure == 'avg':
            out = statistics.mean(measurer(data[col].astype(str)))
        if measure == 'median':
            out = statistics.median(measurer(data[col].astype(str)))
        return out

    def get_topn_freq_bow(data, multi_gram, stem_lemma = '', tag_drop = [],
                          ngram_tag_drop = False, topn = 10):
        """

        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param topn: top n frequency word/bow
        :return: top n frequent bow list
        """
        processed_docs = data.map(
            partial(TextProcessing.find_multiple_gram_tokens,
                    multi_gram = multi_gram,
                    stem_lemma = stem_lemma,
                    tag_drop = tag_drop,
                    ngram_tag_drop = ngram_tag_drop
                    )
        )
        processed_docs = processed_docs.reset_index(drop=True)
        doc_list = []
        for i in range(len(processed_docs)):
            doc_list.extend(processed_docs[i])
        freq_list = nltk.FreqDist(doc_list)
        topn_freq_list = freq_list.most_common(topn)
        return topn_freq_list

    def generate_word_cloud(data, ngram = 1, stem_lemma = 'lemma', tag_drop = [], ngram_tag_drop = False):
        """

        :param ngram:
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :return: show the word cloud plot
        """
        preprocessed_tokens = data.apply(
            lambda x: TextProcessing.find_tokens(example=x, ngram=ngram, stem_lemma=stem_lemma, tag_drop=tag_drop,
                                                 ngram_tag_drop=ngram_tag_drop))
        preprocessed_tokens = preprocessed_tokens.reset_index(drop=True)
        tokens = []
        for i in range(len(preprocessed_tokens)):
            tokens.extend(preprocessed_tokens[i])
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words = ''
        comment_words += " ".join(tokens) + " "
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)
        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

