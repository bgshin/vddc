import numpy as np
import re
import itertools
from collections import Counter



def load_textindex_and_labels(w2vmodel, maxlen, dataset_name, target):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    template_txt = '../data/%s/%s.tsv'
    pathtxt = template_txt % (dataset_name, target)

    x_text_temp = [line.split('\t')[1] for line in open(pathtxt, "r").readlines()]
    # x_text = [s.split(" ") for s in x_text]

    n_vocab = len(w2vmodel.vocab)
    x_text = []
    for s in x_text_temp:
        x_sentence = []
        tokens = s.strip().split(" ")
        n_token = len(tokens)
        for i in range(maxlen):
            if i<n_token:
                token = tokens[i]
            else:
                token = '<(PAD>)'

            try:
                idx = w2vmodel.vocab[token].index

            except:
                idx = n_vocab
                # print token

            x_sentence.append(idx)

        x_text.append(x_sentence)



    y = []

    for line in open(pathtxt, "r").readlines():
        senti=line.split('\t')[0]
        if  senti == '1': # neg
            y.append([1, 0])

        else: # senti == '2': # pos
            y.append([0, 1])

    return [np.array(x_text, dtype=np.int32), np.array(y)]
