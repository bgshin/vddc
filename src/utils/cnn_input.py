"""Routine for decoding the cnnt binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cnn_data_helpers
import numpy as np

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

sequence_length = 1200
embedding_size = 100


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../../data/tw.%s.tfrecords',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DataFeeder(object):
    __metaclass__ = Singleton

    def __init__(self, dataset_name):
        self.sentences, self.labels = cnn_data_helpers.load_data_and_labels(dataset_name=dataset_name, target='sample')
        self.w2v = {'a':np.array([0.1]*100)}

    def _get_word_vector(self, word):
        try:
            vec = self.w2v[word]
        except:
            vec = np.array([0.0]*100)

        return vec

    def get_sample(self):
        return self.sentences[0], self.labels[0]

    def _get_inputs(self, batch_size):
        """
        :param batch_size:
        :return:
        txts: Images. 4D tensor of [batch_size, 1120, 400, 1] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """





def read_cnnt(filename_queue):
    class CNNTRecord(object):
        pass
    result = CNNTRecord()

    label_bytes = NUM_CLASSES*8
    result.maxlen = 60
    result.w2vdim = 400
    input_bytes = (result.maxlen * result.w2vdim)*8
    record_bytes = label_bytes + input_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    value_as_floats = tf.decode_raw(value, tf.float64)
    result.label = tf.cast(value_as_floats[0:3], tf.int32)

    features = value_as_floats[3:3 + record_bytes]
    result.features = tf.reshape(features, [result.maxlen, result.w2vdim])

    return result


def _get_inputs(data_dir, batch_size):
    """Construct distorted input for tweet training using the Reader ops.

    Args:
    data_dir: Path to the tweet  data directory.
    batch_size: Number of images per batch.

    Returns:
    images: Images. 4D tensor of [batch_size, 1120, 400, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
    #              for i in xrange(1, 6)]
    filenames = [data_dir]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cnnt(filename_queue)

    feature = tf.expand_dims(read_input.features, 2)
    read_input.label.set_shape([3])
    label = read_input.label

    feature = tf.cast(feature, tf.float32)
    label = tf.cast(label, tf.float32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CNNT images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    txts, labels = tf.train.batch(
        [feature, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=min_queue_examples + 3 * batch_size)

    return txts, labels

def get_inputs(target, batch_size=4):
    """Construct input for tweet training using the Reader ops.

    Returns:
    images: Images. 4D tensor of [batch_size, 60, 400, 1] size.(4, 60, 400, 1)
    labels: Labels. 1D tensor of [batch_size, 3] size.(4, 3)

    Raises:
    ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir % target
    txts, labels = _get_inputs(data_dir=data_dir, batch_size=batch_size)


    return txts, labels
