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
with Timer('w2v..'):
    w2vmodel, vocab_size = load_w2v(w2vdim=w2vdim)


yelp = DataFeeder('yelp')
txt, y = yelp.get_sample()

cnn = CNN(vocab_size)
# cnn.embedding = np.zeros([vocab_size,w2vdim])
embedding_init = cnn.w2v.assign(cnn.embedding)

sess = tf.Session()
sess.run(embedding_init, feed_dict={cnn.embedding: w2vmodel.syn0})

apple_index = w2vmodel.vocab['apple'].index
print w2vmodel['apple']

# [1,1]

cnn.lookup()
embedded_tokens_expanded = \
    sess.run(cnn.embedded_tokens_expanded, feed_dict={cnn.input_x: np.array([[apple_index]]) }) # apple index 3369

print embedded_tokens_expanded



print txt
print y