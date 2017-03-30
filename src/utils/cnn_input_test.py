from cnn_input import DataFeeder
from src.models.cnn_model import CNN
import numpy as np

yelp = DataFeeder('yelp')
txt, y = yelp.get_sample()

cnn = CNN(100)
cnn.embedding = np.zeros([1200,100])
embedding_init = cnn.w2v.assign(cnn.embedding)

print txt
print y