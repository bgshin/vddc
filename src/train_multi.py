
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from src.utils import cnn_input
from src.models import cnn_model
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.cnn_model import CNN
from src.utils.cnn_input import DataFeeder
from src.utils.word2vecReader import Word2Vec
from src.utils.butils import Timer


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.8, """dropout_keep_prob""")
tf.app.flags.DEFINE_integer('max_batch', 10000,
                            """Maximum possible batch size (56000*1200 = 25Gb, 10000*1200~>5Gb).""")
tf.app.flags.DEFINE_integer('n_dev', 56000,
                            """Number of batches for dev.""")
tf.app.flags.DEFINE_integer('n_tst', 38000,
                            """Number of batches for tst.""")
tf.app.flags.DEFINE_integer('n_trn', 100,
                            """Number of batches for tst.""")

def load_w2v(w2vdim):
    # model_path = '../../data/w2vnew/corpus.friends+nyt+wiki+amazon.fasttext.skip.d%d.vec' % w2vdim
    model_path = '../data/w2v/w2v-%d-amazon.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model, len(model.vocab)


def tower_loss(model, namescope, target):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
    namescope: unique prefix string identifying the tweets tower, e.g. 'tower_0'

    Returns:
     Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for tweets
    # txts, labels = cnnt_input.get_inputs(target, batch_size=batch_size)
    txts, labels = model.lookup()

    # Build inference Graph.
    if target=='trn':
        logits = model.inference(txts, dropout_keep_prob=FLAGS.dropout_keep_prob)

    else:
        logits = model.inference(txts)

    y_true = tf.argmax(labels, 1, name="golds")
    y_pred = tf.argmax(logits, 1, name="predictions")

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _, accuracy = model.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', namescope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss, accuracy, logits, y_true, y_pred


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train():
    maxdev = 0
    maxtst = 0
    maxindex = 0

    with Timer('w2v..'):
        w2vmodel, vocab_size = load_w2v(w2vdim=FLAGS.embedding_size)


    # sess = tf.Session()

    with Timer('loading trn..'):
        yelp_trn = DataFeeder(w2vmodel, 'yelp', 'trn', maxlen=FLAGS.sequence_length, batch_size=FLAGS.n_trn, shuffle=True)

    with Timer('loading dev..'):
        batch_size = min(FLAGS.n_dev, FLAGS.max_batch)
        yelp_dev = DataFeeder(w2vmodel, 'yelp', 'dev', maxlen=FLAGS.sequence_length, batch_size=batch_size, shuffle=False)

    with Timer('loading tst..'):
        batch_size = min(FLAGS.n_tst, FLAGS.max_batch)
        yelp_tst = DataFeeder(w2vmodel, 'yelp', 'tst', maxlen=FLAGS.sequence_length, batch_size=batch_size, shuffle=False)

    """Train cnnt for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        cnn = CNN(vocab_size+1)
        embedding_init = cnn.w2v.assign(cnn.embedding)

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        opt = tf.train.AdamOptimizer(1e-3)

        # Calculate the gradients for each model tower.
        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % 0):
                with tf.name_scope('%s_%d_dev' % (FLAGS.TOWER_NAME, 0)) as namescope:
                    loss_dev, accuracy_dev, logits_dev, y_true_dev, y_pred_dev = \
                        tower_loss(cnn, namescope, 'dev')
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

            with tf.device('/gpu:%d' % 1):
                with tf.name_scope('%s_%d_tst' % (FLAGS.TOWER_NAME, 0)) as namescope:
                    loss_tst, accuracy_tst, logits_tst, y_true_tst, y_pred_tst = \
                        tower_loss(cnn, namescope, 'tst')
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()


            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (FLAGS.TOWER_NAME, i)) as namescope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss, accuracy, _, _, _ = tower_loss(cnn, namescope, 'trn')

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, namescope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            cnn_input.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        gpu_options = tf.GPUOptions(visible_device_list=str('2,3'), allow_growth=True) # o
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        # embedding init with pre-trained weights
        expanded_w2v = np.concatenate((w2vmodel.syn0, np.zeros((1,100))), axis=0)
        sess.run(embedding_init, feed_dict={cnn.embedding: expanded_w2v})

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # _, loss_value = sess.run([train_op, loss])
            x_batch, y_batch = yelp_trn.get_next()
            _, loss_value, accuracy_val = sess.run([train_op, loss, accuracy],
                                                   feed_dict={cnn.input_x: x_batch, cnn.input_y: y_batch})

            # embedded_tokens_expanded = \
            #     sess.run(cnn.embedded_tokens_expanded, feed_dict={cnn.input_x: x})  # apple index 3369

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.4f, acc = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, accuracy_val,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # for i
                x_dev, y_dev = yelp_dev.get_next()
                summary_str, loss_dev_value, accuracy_dev_value, logits_dev_value, y_true_dev_value, y_pred_dev_value = \
                    sess.run([summary_op, loss_dev, accuracy_dev, logits_dev, y_true_dev, y_pred_dev],
                             feed_dict={cnn.input_x: x_dev, cnn.input_y: y_dev})
                summary_writer.add_summary(summary_str, step)

                f1_neg_dev = f1_score(y_true_dev_value==0, y_pred_dev_value==0)
                f1_pos_dev = f1_score(y_true_dev_value == 2, y_pred_dev_value == 2)
                f1_avg_dev = (f1_neg_dev+f1_pos_dev)/2


                format_str = ('[Eval] %s: step %d, loss = %.4f, acc = %.4f, f1neg = %.4f, f1pos = %.4f, f1 = %.4f')
                print(format_str % (datetime.now(), step, loss_dev_value, accuracy_dev_value,
                                    f1_neg_dev, f1_pos_dev, f1_avg_dev))

                x_tst, y_tst = yelp_tst.get_next()

                loss_tst_value, accuracy_tst_value, logits_tst_value, y_true_tst_value, y_pred_tst_value = \
                    sess.run([loss_tst, accuracy_tst, logits_tst, y_true_tst, y_pred_tst],
                             feed_dict={cnn.input_x: x_tst, cnn.input_y: y_tst})

                f1_neg_tst = f1_score(y_true_tst_value == 0, y_pred_tst_value == 0)
                f1_pos_tst = f1_score(y_true_tst_value == 2, y_pred_tst_value == 2)
                f1_avg_tst = (f1_neg_tst + f1_pos_tst) / 2

                format_str = ('[Test] %s: step %d, loss = %.4f, acc = %.4f, f1neg = %.4f, f1pos = %.4f, f1 = %.4f')
                print(format_str % (datetime.now(), step, loss_tst_value, accuracy_tst_value,
                                    f1_neg_tst, f1_pos_tst, f1_avg_tst))

                if maxdev<f1_avg_dev:
                    maxdev = f1_avg_dev
                    maxtst = f1_avg_tst
                    maxindex = step

                format_str = ('[Status] %s: step %d, maxindex = %d, maxdev = %.4f, maxtst = %.4f')
                print(format_str % (datetime.now(), step, maxindex, maxdev, maxtst))


            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
