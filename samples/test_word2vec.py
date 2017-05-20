# -*- coding: utf-8 -*-

import os
import pdb
import gensim

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


# load word2vec model
model = gensim.models.Word2Vec.load(os.path.join(SCRIPT_PATH, 'bgwiki_word2vec'), mmap='r')


# find most similar words
for word, sim in model.most_similar(positive=[u'роднина'], topn=5):
    print('\"%s\"\t- similarity: %g' % (word, sim))

# get coordinates for a particular word
print(model.wv[u'роднина'])
