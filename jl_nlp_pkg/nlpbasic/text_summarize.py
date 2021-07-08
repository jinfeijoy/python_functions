import nltk.data
import math
import pandas as pd
import numpy as np
from numpy.linalg import svd as singular_value_decomposition
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector

def lsa_text_extraction(textdoc, smooth=0.4, MIN_DIMENSIONS = 3, REDUCTION_RATIO = 1/ 1, topn=5):
    """
    reduction_ratio: used to reduce computation cost: limit diagonal size, when it is 1 it keeps original diagonal size, when it is 0.4 only keep 0.4 * original diagonal size
    smooth: is a factor appened to matrix normalization, small value might cause overfitting and large value might cause underfitting
    topn: extract n sentences
    MIN_DIMENSIONS: diagonal minimal size
    """
    ''' document to sentences '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    document = tokenizer.tokenize(textdoc)

    ''' generate term freq matrix '''
    assert 0.0 <= smooth < 1.0
    preprocessed_text = textClean.pipeline(document, multi_gram=[1], lower_case=True, deacc=False, encoding='utf8',
                                           errors='strict', stem_lemma='lemma', tag_drop=[], nltk_stop=True,
                                           stop_word_list=[], check_numbers=False, word_length=2,
                                           remove_consecutives=True)
    dictionary = DocVector.generate_corpus_dict(preprocessed_text, no_below=2, no_above=0.5, keep_n=100000)
    doc_vec = DocVector.create_document_vector(preprocessed_text, dictionary)
    tfmatrix = DocVector.get_vocab_matrix(doc_vec, dictionary)
    matrix_copy = tfmatrix.values.T

    '''
    Computes TF metrics for each sentence (column) in the given matrix and  normalize 
    the tf weights of all terms occurring in a document by the maximum tf in that document 
    according to ntf_{t,d} = a + (1-a)/frac{tf_{t,d}}{tf_{max}(d)^{'}}.

    The smoothing term $a$ damps the contribution of the second term - which may be viewed 
    as a scaling down of tf by the largest tf value in $d$
    '''
    max_word_frequencies = np.max(matrix_copy, axis=0)
    rows, cols = matrix_copy.shape
    for row in range(rows):
        for col in range(cols):
            max_word_frequency = max_word_frequencies[col]
            if max_word_frequency != 0:
                frequency = matrix_copy[row, col] / max_word_frequency
                matrix_copy[row, col] = smooth + (1.0 - smooth) * frequency

    ''' get ranks '''
    u, sigma, v_matrix = singular_value_decomposition(matrix_copy, full_matrices=False)
    assert len(sigma) == v_matrix.shape[0]
    dimensions = max(MIN_DIMENSIONS, int(len(sigma) * REDUCTION_RATIO))
    powered_sigma = tuple(s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma))
    ranks = []
    for column_vector in v_matrix.T:
        rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
        ranks.append(math.sqrt(rank))

    ''' output result '''
    percentile_list = pd.DataFrame(
        {'sentence': document,
         'rank': ranks,
         }).sort_values(by='rank', ascending=False)

    output_sentence = [i for i in percentile_list.head(topn)['sentence']]
    return output_sentence


