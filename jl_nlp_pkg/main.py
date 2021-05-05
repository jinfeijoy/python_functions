# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import gensim
from nlpbasic.TextProcessing import TextProcessing
from nlpbasic.customizedModel import customizedTFIDF
from nlpbasic.customizedModel import customizedLDA
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# print(TextProcessing.find_tokens('you are 2 beautiful'))


raw_data = pd.read_csv("Course_Courseoverviews.csv")

data = raw_data[raw_data.short_description.isnull()==False]
data['doc'] = data[['display_name', 'short_description']].agg('. '.join, axis=1)
data = data.drop(columns = ['course_id', 'last_modified'])
data = data.drop_duplicates()
data['index'] = data.index

if __name__ == '__main__':

    # test = customizedTFIDF.get_top_n_tfidf_bow(data['doc'], [1,2], stem_lemma = '', tag_drop =[],
    #                                            ngram_tag_drop = False, no_below = 5,
    #                                            no_above = 0.5, keep_n = 100000, top_n_tokens = 10)
    lda_all, test1, test2 = customizedLDA.fit_lda(data['doc'], multi_gram=[1,2], no_below=3, tag_drop=[''], top_n_tokens=30, num_topics=5)
    test = customizedLDA.lda_topics(lda_all)
    print(test)
