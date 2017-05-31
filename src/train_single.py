#! /usr/bin/env python
import os
import argparse
import tensorflow as tf
import numpy as np

import datetime
# from utils import cnn_data_helpers
from src.utils.butils import Timer
from sklearn.metrics import precision_score, recall_score, f1_score

# from cnn_models.w2v_cnn import W2V_CNN
from src.models.cnn_model import CNN
from src.utils.cnn_input import DataFeeder

from src.utils.word2vecReader import Word2Vec
import time
import gc
import sys

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_w2v(w2vdim, simple_run=True, source="twitter"):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        if source == "twitter":
            model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
        elif source == "amazon":
            model_path = '../data/emory_w2v/w2v-%d-%s.bin' % (w2vdim, source)

        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

        return model





def run_train(w2vsource, w2vdim, w2vnumfilters, randomseed, l2_reg_lambda):
    with Timer('w2v..'):
        w2vmodel, vocab_size = load_w2v(w2vdim=FLAGS.embedding_size)

    with Timer('loading trn..'):
        yelp_trn = DataFeeder(w2vmodel, 'yelp', 'trn', maxlen=FLAGS.sequence_length, batch_size=FLAGS.n_trn,
                              shuffle=True)

    with Timer('loading dev..'):
        batch_size = min(FLAGS.n_dev, FLAGS.max_batch)
        yelp_dev = DataFeeder(w2vmodel, 'yelp', 'dev', maxlen=FLAGS.sequence_length, batch_size=batch_size,
                              shuffle=False)

    with Timer('loading tst..'):
        batch_size = min(FLAGS.n_tst, FLAGS.max_batch)
        yelp_tst = DataFeeder(w2vmodel, 'yelp', 'tst', maxlen=FLAGS.sequence_length, batch_size=batch_size,
                              shuffle=False)

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))

    cnn = CNN(vocab_size + 1)
    embedding_init = cnn.w2v.assign(cnn.embedding)

    sess.run(init)
    # embedding init with pre-trained weights
    expanded_w2v = np.concatenate((w2vmodel.syn0, np.zeros((1, 100))), axis=0)
    sess.run(embedding_init, feed_dict={cnn.embedding: expanded_w2v})

    with tf.Graph().as_default():
        max_af1_dev = 0
        index_at_max_af1_dev = 0
        af1_tst_at_max_af1_dev = 0

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        # gpu_options = tf.GPUOptions(visible_device_list=str('0,1,2,3'), allow_growth=True) # o
        gpu_options = tf.GPUOptions(visible_device_list=str('3'), allow_growth=True)  # o
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2vsource', default='twitter', choices=['twitter', 'amazon'], type=str)
    parser.add_argument('--w2vdim', default=100, type=int)
    parser.add_argument('--w2vnumfilters', default=64, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--model', default='w2v', choices=['w2v'],
                        type=str)  # w2v, w2vlex, attention
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--l2_reg_lambda', default=2.0, type=float)
    parser.add_argument('--l1_reg_lambda', default=0.0, type=float)

    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])

    print 'ADDITIONAL PARAMETER\n w2vsource: %s\n w2vdim: %d\n w2vnumfilters: %d\n ' \
          'randomseed: %d\n num_epochs: %d\n' \
          'l2_reg_lambda: %f\n l2_reg_lambda: %f\n' \
          % (args.w2vsource, args.w2vdim, args.w2vnumfilters, args.randomseed,args.num_epochs,
             args.l2_reg_lambda, args.l1_reg_lambda)

    with Timer('trn..'):
        run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.randomseed,
                  args.l2_reg_lambda)