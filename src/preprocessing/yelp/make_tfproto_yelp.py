from src.utils.butils import Timer
import numpy as np
import tensorflow as tf
import os
from src.utils.word2vecReader import Word2Vec
from src.utils import cnn_data_helpers
from tqdm import tqdm

w2vdim = 100
max_len = 1200

def load_w2v(w2vdim):
    model_path = '../../../data/w2v/w2v-%d-amazon.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model


def _record(self, label, num):
    input_size = max_len * w2vdim
    x = np.ones([max_len ,w2vdim]) * num
    x = x.reshape([input_size])
    x = np.append(float(label), x)
    # x = np.array([label] + [num] * input_size)
    record = x.tostring()
    expected = np.array([[[num]] * w2vdim] * max_len)
    return record, expected[: ,: ,0]


def write_record(target):
    tfrecords_filename = '../../../data/yelp/yelp.%s.tfrecords' % target
    # writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    x_train, y_train = cnn_data_helpers.load_data_new('yelp', target, w2vmodel, max_len)

    records = []
    with open(tfrecords_filename, "wb") as writer:
        for idx in tqdm(range(len(x_train))):
            x = x_train[idx]
            y = y_train[idx]
            # print x.shape
            # print y.shape

            input_size = max_len* w2vdim
            x = x.reshape([input_size])
            all = np.append(map(float, y), x)
            record = all.tostring()
            records.append(record)
            contents = b"".join(records)

        writer.write(contents)

with Timer('w2v..'):
    w2vmodel = load_w2v(w2vdim)


# targets = ['dev', 'tst', 'trn']
targets = ['dev']

for t in targets:
    with Timer('writing_%s..' % t):
        write_record(t)
