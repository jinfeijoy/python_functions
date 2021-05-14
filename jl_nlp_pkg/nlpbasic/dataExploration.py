import numpy as np
import statistics
import nltk
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

    def get_topn_freq_bow(corpus, topn = 10):
        """
        :param corpus: corpus generated from doc_tokenize()
        :return: top n frequent bow list
        """

        processed_docs = corpus
        doc_list = []
        for i in range(len(processed_docs)):
            doc_list.extend(processed_docs[i])
        freq_list = nltk.FreqDist(doc_list)
        topn_freq_list = freq_list.most_common(topn)
        return topn_freq_list

    def generate_word_cloud(corpus):
        """

        :param corpus: corpus generated from doc_tokenize()
        :return: show the word cloud plot
        """
        preprocessed_tokens = corpus
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



