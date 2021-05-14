import string
import gensim
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

class TextProcessing(object):

    def __init__(self):
        pass

    def get_wordnet_pos(word):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ, "N": wordnet.NOUN,
            "V": wordnet.VERB, "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def stem_lemma_process(word, stem_lemma = ''):
        porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        if stem_lemma not in ('','stem','lemma'):
            raise TypeError('stem_lemma should be either "stem" or "lemma" or ""')
        if stem_lemma == '':
            word_output = word
        if stem_lemma == 'stem':
            word_output = porter.stem(word)
        if stem_lemma == 'lemma':
            word_output = lemmatizer.lemmatize(word, pos = TextProcessing.get_wordnet_pos(word))
        return word_output


    def find_tokens(example, ngram = 1, stem_lemma = '', tag_drop = [], ngram_tag_drop = False, user_defined_stopwords = []):
        '''
        this function is to generate n-gram tokens given a sentence/document
        :param ngram: n-gram to included in tokens, e.g. 1, 2
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param user_defined_stopwords: user defined stopwords
        :return: return a list of tokens
        :ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        '''
        example = example.lower().translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('english'))
        filtered_sentence = []
        tmp_sentence = []
        if ngram == 1:
            word_tokens = [t[0] for t in pos_tag(word_tokenize(example)) if t[1][0] not in tag_drop]
            for w in word_tokens:
                if w not in stop_words and w not in user_defined_stopwords:
                    if w.isdigit() == False:
                        filtered_sentence.append(TextProcessing.stem_lemma_process(w, stem_lemma = stem_lemma))
        else:
            if ngram_tag_drop == True:
                word_tokens = [t[0] for t in pos_tag(word_tokenize(example)) if t[1][0] not in tag_drop]
            if ngram_tag_drop == False:
                word_tokens = word_tokenize(example)
            for w in word_tokens:
                if w not in stop_words and w not in user_defined_stopwords:
                    if w.isdigit() == False:
                        tmp_sentence.append(TextProcessing.stem_lemma_process(w, stem_lemma = stem_lemma))
            n_grams = ngrams(tmp_sentence, ngram)
            for grams in n_grams:
                filtered_sentence.append(" ".join(grams))
        return filtered_sentence

    def find_multiple_gram_tokens(examples, multi_gram, stem_lemma = '', tag_drop = [], ngram_tag_drop = False, user_defined_stopwords = []):
        '''
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param user_defined_stopwords: user defined stopwords
        :return: return a list of tokens
       '''
        multi_tokens = []
        for i in multi_gram:
            multi_tmp_tokens = TextProcessing.find_tokens(examples, i, stem_lemma, tag_drop,
                                                          ngram_tag_drop = ngram_tag_drop, user_defined_stopwords = user_defined_stopwords)
            multi_tokens.extend(multi_tmp_tokens)
        return multi_tokens

    def keep_specific_tokens(corpus, selected_tokens = []):
        """
        :param corpus: tokens generated from multi_gram
        :param selected_tokens: list of tokens we want to keep in "examples"
        :return: tokens
        """
        return [i for i in corpus if i in selected_tokens]

    def doc_tokenize(data, multi_gram, stem_lemma = '', tag_drop = [], ngram_tag_drop = False,
                     user_defined_stopwords = [] ,keep_tokens = []):
        """
        :param data: data column, eg. data['review']
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param user_defined_stopwords: user defined stopwords
        :param keep_tokens: list of tokens we want to keep in "examples"
        :return: return a list of tokens
        """
        if len(keep_tokens) == 0:
            return [list(TextProcessing.find_multiple_gram_tokens(doc, multi_gram, stem_lemma = stem_lemma,
                                                                  tag_drop = tag_drop, ngram_tag_drop = ngram_tag_drop,
                                                                  user_defined_stopwords = user_defined_stopwords)
                         ) for doc in data]
        else:
            return [list(TextProcessing.keep_specific_tokens(TextProcessing.find_multiple_gram_tokens(doc, multi_gram,
                                                                                                      stem_lemma=stem_lemma,
                                                                                                      tag_drop=tag_drop,
                                                                                                      ngram_tag_drop=ngram_tag_drop,
                                                                                                      user_defined_stopwords = user_defined_stopwords),
                                                             selected_tokens=keep_tokens)
                         ) for doc in data]



class DocVector(object):
    def __init__(self):
        pass

    def get_vocab(tokens):
        '''
        this function is to get vocabulary
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
        :param corpus: corpus generated from doc_tokenize()
        :param no_below: filter out tokens that less than no_below documents (absolute number)
        :param no_above: filter out tokens that more than no_above documents (fraction of total corpus size, not absolute number).
        :param keep_n: filter out tokens that after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
        :return: bow corpus, dictionary
        """
        dictionary = gensim.corpora.Dictionary(corpus)
        dictionary.filter_extremes(no_below=no_below,
                                   no_above=no_above,
                                   keep_n=keep_n)
        bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
        return bow_corpus, dictionary

    def get_vocab_matrix(document_vector, vocabulary):
        """
        :param document_vector: bow_corpus generated from generate_corpus_dict
        :param vocabulary: dictionary generated from generate_corpus_dict
        :return: term document matrix
        """
        my_matrix = pd.DataFrame(0.0, index=np.arange(len(document_vector)), columns=[i for i in vocabulary.values()])
        for i in range(len(document_vector)):
            for j in range(len(document_vector[i])):
                my_matrix.at[i, vocabulary[document_vector[i][j][0]]] = document_vector[i][j][1]
        my_matrix.index.name = 'doc'
        return my_matrix