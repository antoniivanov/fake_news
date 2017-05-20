import pdb
import gensim
import pandas as pd
import numpy as np
import gensim.utils as utils

import logging

def tokenize(content):
    return [
        utils.to_unicode(token) for token in utils.tokenize(content, lower=True, errors='ignore')
        if 2 <= len(token) <= 15 and not token.startswith('_')
    ]

def embed_content(data, col_name, output_arr):
    unmatched_words = 0
    skipped_docs = 0
    total_words = 0

    for i, row in data.iterrows():
        rowdata = row[col_name]
        try:
            tokenized = tokenize(rowdata)
        except:
            skipped_docs += 1
            continue

        for j, word in enumerate(tokenized):
            total_words += 1
            try:
                output_arr[i, j] = w2v.wv[word]
            except KeyError:
                output_arr[i, j] = UNK
                unmatched_words += 1

    return unmatched_words, total_words, skipped_docs

# dimensionality of w2v embeddings
word2vec_dim = 100

# some constants
# TODO: should be read at runtime once we switch to the real CSV
max_body_len = 5337
max_title_len = 41

# setup logger
logger = logging.getLogger('fakenews_logger')
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

logger.info('loading model')
w2v = gensim.models.Word2Vec.load('bgwiki_word2vec_news_100', mmap='r')

logger.info('loading csv data')
data = pd.read_csv('FN_Training_Set.csv', encoding='windows-1251')

UNK = np.zeros((word2vec_dim,), dtype=np.float)
output_titles = np.zeros((data.shape[0], max_title_len, word2vec_dim), dtype=np.float)
output_content = np.zeros((data.shape[0], max_body_len, word2vec_dim), dtype=np.float)

# embed all titles
logger.info('embedding all titles')
missing_words, total_words, skipped_docs = embed_content(data, u'Content Title', output_titles)

logger.info('[titles] {0} words could NOT be matched in the w2v vocabulary, Total number of words: {1}. Skipped documents: {2}'.format(missing_words, total_words, skipped_docs))

logger.info('persisting embedded titles')
np.save('embedded_titles_100', output_titles)

# embed all content
logger.info('embedding all content')
missing_words, total_words, skipped_docs = embed_content(data, u'Content', output_content)

logger.info('[content] {0} words could NOT be matched in the w2v vocabulary, Total number of words: {1}. Skipped documents: {2}'.format(missing_words, total_words, skipped_docs))

logger.info('persisting embedded content')
np.save('embedded_content_100', output_content)
