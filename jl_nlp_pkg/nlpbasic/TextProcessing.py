import string
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


    def find_tokens(example, ngram = 1, stem_lemma = '', tag_drop = [], ngram_tag_drop = False):
        '''
        this function is to generate n-gram tokens given a sentence/document
        :param ngram: n-gram to included in tokens, e.g. 1, 2
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
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
                if w not in stop_words:
                    if w.isdigit() == False:
                        filtered_sentence.append(TextProcessing.stem_lemma_process(w, stem_lemma = stem_lemma))
        else:
            if ngram_tag_drop == True:
                word_tokens = [t[0] for t in pos_tag(word_tokenize(example)) if t[1][0] not in tag_drop]
            if ngram_tag_drop == False:
                word_tokens = word_tokenize(example)
            for w in word_tokens:
                if w not in stop_words:
                    if w.isdigit() == False:
                        tmp_sentence.append(TextProcessing.stem_lemma_process(w, stem_lemma = stem_lemma))
            n_grams = ngrams(tmp_sentence, ngram)
            for grams in n_grams:
                filtered_sentence.append(" ".join(grams))
        return filtered_sentence

    def find_multiple_gram_tokens(examples, multi_gram, stem_lemma = '', tag_drop = [], ngram_tag_drop = False):
        '''
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :return: return a list of tokens
       '''
        multi_tokens = []
        for i in multi_gram:
            multi_tmp_tokens = TextProcessing.find_tokens(examples, i, stem_lemma, tag_drop, ngram_tag_drop = ngram_tag_drop)
            multi_tokens.extend(multi_tmp_tokens)
        return multi_tokens

    def keep_specific_tokens(examples, multi_gram, stem_lemma = '', tag_drop = [], ngram_tag_drop = False, selected_tokens = []):
        """
        :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
        :param stem_lemma: do stemmer or lemmatizer or nothing processing, e.g. 'stem', 'lemma', ''
        :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
        :param ngram_tag_drop: True: drop words with specific tag in n-gram, False: keep all words when generate n-gram tokens
        :param selected_tokens: list of tokens we want to keep in "examples"
        :return: tokens
        """
        tmp_tokens = TextProcessing.find_multiple_gram_tokens(examples, multi_gram, stem_lemma, tag_drop, ngram_tag_drop)
        tmp_tokens = [i for i in tmp_tokens if i in selected_tokens]
        return tmp_tokens

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

