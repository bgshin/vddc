"""Routine for decoding the cnnt binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from src.utils import cnn_data_helpers
import numpy as np
import math

# Global constants describing the CIFAR-10 data set.
# NUM_CLASSES = 2
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('sequence_length', 1200,
#                             """Number of batches to run.""")
# tf.app.flags.DEFINE_integer('embedding_size', 1200,
#                             """Number of batches to run.""")

# FLAGS.sequence_length = 1200
# FLAGS.embedding_size = 100


# FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../../data/tw.%s.tfrecords',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")

class DataFeeder(object):
    # __metaclass__ = Singleton

    def __init__(self, w2vmodel, dataset_name, target='sample', maxlen=10, batch_size = None, shuffle=True):
        self.ids, self.labels = cnn_data_helpers.load_textindex_and_labels(w2vmodel=w2vmodel, maxlen=maxlen,
                                                                            dataset_name=dataset_name, target=target)
        self.n = len(self.labels)
        self.shuffle = shuffle

        if batch_size == None:
            self.batch_size = self.n
        else:
            self.batch_size = min(self.n, batch_size)

        self.set_batch(self.batch_size)


    def set_batch(self, batch_size=0):
        self.batch_size = min(self.n, batch_size)

        if self.batch_size == 0: # all samples
            self.num_batches_per_epoch = self.n

        else:
            self.num_batches_per_epoch = math.ceil(self.n*1.0 / self.batch_size) + 1

        np.random.seed(3)  # FIX RANDOM SEED
        self.batch_init()

    def batch_init(self):
        if self.shuffle:
            self.shuffle_indices = np.random.permutation(np.arange(self.n))

        else:
            self.shuffle_indices = np.arange(self.n)

        self.shuffled_ids = self.ids[self.shuffle_indices]
        self.shuffled_labels = self.labels[self.shuffle_indices]

        self.batch_num = 0
        self.start_index = self.batch_num * self.batch_size
        self.end_index = min((self.batch_num + 1) * self.batch_size, self.n)

    def set_batch_all(self):
        self.set_batch(self.n)

    def get_sample(self):
        return self.ids[0], self.labels[0]

    def get_next(self):
        index_log = '%d:%d' % (self.start_index, self.end_index)
        ids_batch = self.shuffled_ids[self.start_index:self.end_index]
        labels_batch = self.shuffled_labels[self.start_index:self.end_index]
        self.batch_num += 1

        if self.batch_num * self.batch_size>=self.n:
            self.batch_init()

        else:
            self.start_index = self.batch_num * self.batch_size
            self.end_index = min((self.batch_num + 1) * self.batch_size, self.n)

        return ids_batch, labels_batch




