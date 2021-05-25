from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.utils import tokenize
from nltk.corpus import wordnet
from nltk import ngrams
from itertools import groupby
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import re

def get_hashtag(examples):
    return re.findall(r"#(\w+)", examples)

def remove_string_startwith(text_data, txt_start):
    return re.sub(r"{}\S+".format(txt_start), "", text_data)

def doc_tokenize(examples, ngram=1, nltk_stop = True, stop_word_list = [], remove_pattern = [],
                 lower_case = False, deacc = False, encoding = 'utf8', errors = 'strict'):
    """
    :param examples: sentence
    :param ngram: number of gram bag of words
    :param nltk_stop: Ture or False, to remove nltk stop words
    :param stop_word_list: user defined stop words list
    :param remove_pattern: to remove string after [""], default value is []
    :param lower_case: lowercase the input string or not
    :param deacc: (bool, optional) -- Remove accentuation using decaccent
    :param encoding:
    :param errors:
    :return: list of lists of tokenized documents
    """
    if len(remove_pattern) == 0:
        examples = examples
    elif len(remove_pattern) > 0:
         for i in range(0, len(remove_pattern)):
            examples = remove_string_startwith(examples, remove_pattern[i])


    stop_words = []
    if nltk_stop:
        stop_words = list(stopwords.words('english'))
    if not isinstance(stop_word_list, list):
        raise TypeError('Input should be a list of stopwords')
    stop_words.extend(stop_word_list)
    # if len(stop_words) < 1:
    #     print("No Stopwords Provided!")

    filtered_sentence = []
    if ngram == 1:
        tokens = tokenize(examples, lower = lower_case, deacc = deacc, encoding=encoding, errors=errors)
        for token in tokens:
            if token not in stop_words:
                filtered_sentence.append(token)
    if ngram > 1:
        tmp_sentence = []
        tokens = tokenize(examples, lower=lower_case, deacc=deacc, encoding=encoding, errors=errors)
        for token in tokens:
            if token not in stop_words:
                tmp_sentence.append(token)
        n_grams = ngrams(tmp_sentence, ngram)
        for grams in n_grams:
            filtered_sentence.append(" ".join(grams))
    return filtered_sentence



def doc_tokenize_multi_gram(corpus, multi_gram, nltk_stop = True, stop_word_list = [], remove_pattern = [],
                            lower_case = False, deacc = False, encoding = 'utf8', errors = 'strict'):
    '''
    :param corpus: list of strings -- input list of documents
    :param multi_gram: multiple gram list, e.g. [1,2,3] indicate to output 1 2 and 3 grams tokens
    :param nltk_stop: Ture or False, to remove nltk stop words
    :param stop_word_list: user defined stop words list
    :param remove_pattern: to remove string after [""], default value is []
    :param lower_case: lowercase the input string or not
    :param deacc: (bool, optional) -- Remove accentuation using decaccent
    :param encoding:
    :param errors:
    :return: return a list of list of tokens
   '''
    if not isinstance(corpus, list):
        raise TypeError('Input should be a list of strings!')
    if not isinstance(multi_gram, list):
        raise TypeError('multi_gram input should be a list of int!')
    output = []
    for doc in corpus:
        multi_tokens = []
        for ngram in multi_gram:
            multi_tmp_tokens = doc_tokenize(examples=doc, ngram=ngram, nltk_stop = nltk_stop, stop_word_list = stop_word_list,
                                            remove_pattern=remove_pattern,
                                            lower_case = lower_case, deacc = deacc, encoding = encoding, errors=errors)
            multi_tokens.extend(multi_tmp_tokens)
        output.append(multi_tokens)
    return output

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ, "N": wordnet.NOUN,
        "V": wordnet.VERB, "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def remove_pos_token(tokenized_corpus, tag_drop = []):
    """
    :param tokenized_corpus: a list of lists of tokens
    :param tag_drop: whether to drop word with specific tag: J--objective, N--noun, V--verb, R--adv. e.g. ['J'], ['J','N']
    :return:
    """
    return [[t[0] for t in pos_tag(doc) if t[1][0] not in tag_drop] for doc in tokenized_corpus]


def stem_lemma_process(tokenized_corpus, stem_lemma = ''):
    """
    :param tokenized_corpus: a list of lists of tokens
    :param stem_lemma: function to stem/lemma documents, should be '' or 'stem' or 'lemma'
    :return:
    """
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    if stem_lemma not in ('','stem','lemma'):
        raise TypeError('stem_lemma should be either "stem" or "lemma" or ""')
    if stem_lemma == '':
        output = tokenized_corpus
    if stem_lemma == 'stem':
        output = [[porter.stem(token) for token in doc] for doc in tokenized_corpus]
    if stem_lemma == 'lemma':
        output = [[lemmatizer.lemmatize(token, pos = get_wordnet_pos(token)) for token in doc] for doc in tokenized_corpus]
    return output

def remove_invalid_tokens(tokenzied_corpus,
                          check_numbers = True, word_length = 0, remove_consecutives = False):
    """
    :param tokenized_corpus: a list of lists of tokens
    :param check_numbers: Ture or False to remove numbers from tokens
    :param word_length: integer, to filter out word with len(word) less than word_length
    :param remove_consecutives: remove consecutive occurrences of some word
    :return: a list of list of tokens
    """

    numbers = ['zero','one','two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
               'eleven', 'twelve', 'third', 'twent', 'thirt', 'fort', 'fift', 'ninth', 'hundred']
    def is_number(token):
        for num in numbers:
            if (num == token) or (num in token):
                return True
        return False

    def is_valid(token):
        if (len(token)<word_length):
            return -1
        elif check_numbers and (is_number(token) or str(token).isdigit()):
            return -1
        # elif (token in stop_words):
        #     return -1
        return token

    def clean_document(token_list):
        clean_list = [is_valid(token) for token in token_list]
        clean_list = list(filter((-1).__ne__, clean_list))
        #remove consecutive occurences of same word
        if remove_consecutives:
            clean_list = [x[0] for x in groupby(clean_list)]
        return clean_list

    return [clean_document(doc) for doc in tokenzied_corpus]

def pipeline(corpus, multi_gram= [1], nltk_stop = True, stop_word_list = [], remove_pattern = [],
             lower_case = False, deacc = False, encoding = 'utf8', errors = 'strict',
             stem_lemma = '', tag_drop = [], check_numbers = True, word_length = 0,
             remove_consecutives = False):
    """
    doc_tokenize parameters:
    :param corpus: list of strings -- input list of documents
    :param ngram: number of gram bag of words
    :param nltk_stop: Ture or False, to remove nltk stop words
    :param stop_word_list: user defined stop words list
    :param remove_pattern: to remove string after [""], default value is []
    :param lower_case: lowercase the input string or not
    :param deacc: (bool, optional) -- Remove accentuation using decaccent
    :param encoding:
    :param errors:

    stem_lemma_process parameters:
    :param stem_lemma: function to stem/lemma documents, should be '' or 'stem' or 'lemma'

    remove_invalid_tokens parameters:
    :param check_numbers: Ture or False to remove numbers from tokens
    :param word_length: integer, to filter out word with len(word) less than word_length
    :param remove_consecutives: remove consecutive occurrences of some word

    :return: the text according to the set flags
    """
    if not isinstance(corpus, list):
        raise TypeError('Input should be a list of strings')

    result = doc_tokenize_multi_gram(corpus, multi_gram=multi_gram, nltk_stop = nltk_stop, stop_word_list = stop_word_list,
                                     remove_pattern = remove_pattern,
                                     lower_case = lower_case, deacc = deacc, encoding = encoding, errors = errors)
    result = stem_lemma_process(result, stem_lemma = stem_lemma)
    result = remove_pos_token(result, tag_drop = tag_drop)
    result = remove_invalid_tokens(result, check_numbers=check_numbers, word_length=word_length,
                                    remove_consecutives=remove_consecutives)
    return result

def keep_specific_tokens(corpus, selected_tokens = []):
    """
    :param corpus: tokens generated from multi_gram
    :param selected_tokens: list of tokens we want to keep in "examples"
        :return: tokens
        """
    return [i for i in corpus if i in selected_tokens]