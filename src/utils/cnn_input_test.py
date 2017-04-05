from src.utils.butils import Timer
from cnn_input import DataFeeder
from src.models.cnn_model import CNN
import numpy as np
import tensorflow as tf

from src.utils.word2vecReader import Word2Vec
from tqdm import tqdm

def load_w2v(w2vdim):
    # model_path = '../../data/w2vnew/corpus.friends+nyt+wiki+amazon.fasttext.skip.d%d.vec' % w2vdim
    model_path = '../../data/w2v/w2v-%d-amazon.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model, len(model.vocab)

w2vdim=100
maxlen=10
with Timer('w2v..'):
    w2vmodel, vocab_size = load_w2v(w2vdim=w2vdim)


yelp_dev = DataFeeder(w2vmodel, 'yelp', 'sample', maxlen=maxlen, batch_size=4, shuffle=False)
yelp_trn = DataFeeder(w2vmodel, 'yelp', 'sample', maxlen=maxlen)
yelp_tst = DataFeeder(w2vmodel, 'yelp', 'sample', maxlen=maxlen)


print 'batch 4'
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log

print 'batch all'
yelp_dev.set_batch_all()
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log

print 'batch 2'
yelp_dev.set_batch(2)
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log

print 'batch 9'
yelp_dev.set_batch(9)
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log


print 'batch 12'
yelp_dev.set_batch(12)
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log

print 'batch 3'
yelp_dev.set_batch(3)
for i in range(10):
    x,y, index_log = yelp_dev.get_next()
    print len(x), index_log


# exit(1)

cnn = CNN(vocab_size)
embedding_init = cnn.w2v.assign(cnn.embedding)
sess = tf.Session()
sess.run(embedding_init, feed_dict={cnn.embedding: w2vmodel.syn0})

# apple_index = w2vmodel.vocab['apple'].index
print w2vmodel['Wow']
print w2vmodel[',']
print w2vmodel['really']

# [1,1]
yelp_dev.set_batch(3)
x,y, index_log = yelp_dev.get_next()

cnn.lookup()
embedded_tokens_expanded = \
    sess.run(cnn.embedded_tokens_expanded, feed_dict={cnn.input_x: x }) # apple index 3369

print embedded_tokens_expanded



# print txt
# print y