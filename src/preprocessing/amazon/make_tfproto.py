from src.utils.butils import Timer
import numpy as np
import tensorflow as tf
import os
from src.utils.word2vecReader import Word2Vec
from src.utils import cnn_data_helpers
from tqdm import tqdm

def load_w2v(w2vdim):
    model_path = '../../data/emory_w2v/w2v-%d.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model


def _record(self, label, num):
    input_size = 60 * 400
    x = np.ones([60 ,400]) * num
    x = x.reshape([input_size])
    x = np.append(float(label), x)
    # x = np.array([label] + [num] * input_size)
    record = x.tostring()
    expected = np.array([[[num]] * 400] * 60)
    return record, expected[: ,: ,0]

def testSimple(self):
    labels = [1, 0, 2]
    records = [self._record(labels[0], 0.1),
               self._record(labels[1], 0.2),
               self._record(labels[2], 0.3)]
    contents = b"".join([record for record, _ in records])
    expected = [expected for _, expected in records]
    filename = os.path.join(self.get_temp_dir(), "cnnt")
    open(filename, "wb").write(contents)

def write_record(target):
    tfrecords_filename = '../../data/tw.%s.tfrecords' % target
    # writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    x_train, y_train = cnn_data_helpers.load_data_new(target, w2vmodel, max_len)

    records = []
    with open(tfrecords_filename, "wb") as writer:
        for idx in tqdm(range(len(x_train))):
            x = x_train[idx]
            y = y_train[idx]
            # print x.shape
            # print y.shape

            input_size = 60 * 400
            x = x.reshape([input_size])
            all = np.append(map(float, y), x)
            record = all.tostring()
            records.append(record)
            contents = b"".join(records)

        writer.write(contents)

w2vdim = 400
max_len = 60
with Timer('w2v..'):
    w2vmodel = load_w2v(w2vdim)


# targets = ['dev', 'tst', 'trn']
targets = ['dev']

for t in targets:
    with Timer('writing_%s..' % t):
        write_record(t)
