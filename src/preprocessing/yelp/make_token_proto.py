from src.utils.butils import Timer
import numpy as np
import os
from src.utils import cnn_data_helpers
from tqdm import tqdm

max_len = 1200


def write_record(target):
    tfrecords_filename = '../../../data/yelp.%s.tfrecords' % target
    sentences, labels = cnn_data_helpers.load_data_and_labels(dataset_name='yelp', target=target)
    txts = cnn_data_helpers.pad_sentences(sentences, max_len)

    records = []
    with open(tfrecords_filename, "wb") as writer:
        for idx in tqdm(range(len(labels))):
            x = txts[idx]
            y = labels[idx]
            # print x.shape
            # print y.shape

            input_size = 60 * 400
            x = x.reshape([input_size])
            all = np.append(map(float, y), x)
            record = all.tostring()
            records.append(record)
            contents = b"".join(records)

        writer.write(contents)

# w2vdim = 400
# max_len = 60


# targets = ['dev', 'tst', 'trn']
targets = ['sample']

for t in targets:
    with Timer('writing_%s..' % t):
        write_record(t)
