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
    max_len = 60
    num_classes = 3

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

    cnn = CNN(vocab_size + 1)
    embedding_init = cnn.w2v.assign(cnn.embedding)

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

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        f1_summary = tf.summary.scalar("avg_f1", cnn.avg_f1)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.variables_initializer())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
            #      format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, score_type='f1', writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{} : {} step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format("DEV", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            if writer:
                writer.add_summary(summaries, step)

            if score_type == 'f1':
                return avg_f1
            else:
                return accuracy

        def test_step(x_batch, y_batch, score_type='f1', writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{} : {} step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format("TEST", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            if writer:
                writer.add_summary(summaries, step)

            if score_type == 'f1':
                return avg_f1
            else:
                return accuracy

        # Generate batches
        batches = cnn_data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("Evaluation:")
                score_type = 'f1'
                curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type=score_type)
                curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer, score_type=score_type)

                if curr_af1_dev > max_af1_dev:
                    max_af1_dev = curr_af1_dev
                    index_at_max_af1_dev = current_step
                    af1_tst_at_max_af1_dev = curr_af1_tst

                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


                print 'Status: [%d] Max f1 for dev (%f), Max f1 for tst (%f)\n' % (
                    index_at_max_af1_dev, max_af1_dev, af1_tst_at_max_af1_dev)


                sys.stdout.flush()



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