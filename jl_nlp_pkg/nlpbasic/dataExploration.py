import numpy as np
import statistics
from nlpbasic.TextProcessing import TextProcessing

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

    