import tensorflow as tf
import re


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('sequence_length', 1200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('embedding_size', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('NUM_CLASSES', 2,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_filters', 64,
                            """Number of batches to run.""")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# FLAGS.num_filters
tf.flags.DEFINE_string("TOWER_NAME", "tower", "multi gpu tower name")

# FLAGS.TOWER_NAME = 'tower'

def _variable_on_cpu(name, shape, initializer, trainable):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """

    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, trainable):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
        trainable)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def _variable_with_weight_decay_xavier(name, shape, wd, trainable):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer(),
        trainable)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % FLAGS.TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


class CNN(object):
    def __init__(self, vocab_size):
        self.input_x = tf.placeholder(tf.int32, [None, FLAGS.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, FLAGS.NUM_CLASSES], name="input_y")
        self.embedding = tf.placeholder(tf.float32, [vocab_size, FLAGS.embedding_size])
        print 'self.embedding', self.embedding

        self.w2v = _variable_with_weight_decay('embedding',
                                                shape=[vocab_size, FLAGS.embedding_size],
                                                stddev=0.1,
                                                wd=None,
                                                trainable=False)
        print 'self.w2v', self.w2v

        self.embedding_params = self.w2v.assign(self.embedding)
        print 'self.embedding_params',self.embedding_params

        # embedding_init = self.w2v.assign(self.embedding)

    def lookup(self):
        # embedded_tokens = tf.nn.embedding_lookup(self.embedding_params, self.input_x)
        embedded_tokens = tf.nn.embedding_lookup(self.w2v, self.input_x)
        print 'embedded_tokens', embedded_tokens
        self.embedded_tokens_expanded = tf.expand_dims(embedded_tokens, -1)
        # (?, 1200, 100, 1)
        print 'embedded_tokens_expanded', self.embedded_tokens_expanded

        return self.embedded_tokens_expanded, self.input_y





    def inference(self, txts, dropout_keep_prob=1.0):
        """Build the cnn based sentiment prediction model.

        Args:
        txts: text returned from get_inputs().

        Returns:
        Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #


        pooled_outputs = []
        for i, filter_size in enumerate(list(map(int, FLAGS.filter_sizes.split(",")))):
            with tf.variable_scope("conv-maxpool-%s" % filter_size) as scope:
                cnn_shape = [filter_size, FLAGS.embedding_size, 1, FLAGS.num_filters]
                kernel = _variable_with_weight_decay('weights',
                                                     shape=cnn_shape,
                                                     stddev=0.1,
                                                     wd=None,
                                                     trainable=True)
                conv = tf.nn.conv2d(txts, kernel, [1, 1, 1, 1], padding='VALID')
                biases = _variable_on_cpu('biases', [FLAGS.num_filters], tf.constant_initializer(0.0), trainable=True)
                pre_activation = tf.nn.bias_add(conv, biases)
                conv_out = tf.nn.relu(pre_activation, name=scope.name)
                _activation_summary(conv_out)


                ksize = [1, FLAGS.sequence_length - filter_size + 1, 1, 1]
                print 'filter_size', filter_size
                print 'ksize', ksize
                print 'conv_out', conv_out
                pooled = tf.nn.max_pool(conv_out, ksize=ksize, strides=[1, 1, 1, 1],
                                     padding='VALID', name='pool1')

                norm_pooled = tf.nn.lrn(pooled, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                name='norm1')

                # pooled_outputs.append(pooled)
                pooled_outputs.append(norm_pooled)




        # print 'norm1', norm1
        num_filters_total = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(","))))
        h_pool = tf.concat(pooled_outputs, 3)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print 'h_pool', h_pool
        print 'h_pool_flat', h_pool_flat

        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)


        # num_filters_total = num_filters * 1
        # norm_flat = tf.reshape(norm1, [-1, num_filters_total])


        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay_xavier('weights', [num_filters_total, FLAGS.NUM_CLASSES],
                                                wd=0.2, trainable=True)
            biases = _variable_on_cpu('biases', [FLAGS.NUM_CLASSES],
                                      tf.constant_initializer(0.1), trainable=True)
            softmax_linear = tf.add(tf.matmul(h_pool_flat, weights), biases, name=scope.name)
            _activation_summary(softmax_linear)

        return softmax_linear

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.

        Add summary for "Loss" and "Loss/avg".
        Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

        Returns:
        Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        # labels = tf.cast(labels, tf.int64)
        # labels = tf.cast(tf.argmax(labels, 1), tf.int64)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        golds = tf.argmax(labels, 1, name="golds")
        predictions = tf.argmax(logits, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy
