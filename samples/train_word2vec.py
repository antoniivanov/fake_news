import os
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.corpora import Dictionary

import logging

dictionary_name = 'bgwiki_wordids.txt.bz2'
dump_name = 'bgwiki-latest-pages-articles.xml.bz2'

if __name__ == '__main__':

    logger = logging.getLogger('w2v_logger')

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    logger.info("building dictionary")
    if not os.path.isfile(dictionary_name):
        logging.info('Dictionary has not been created yet..')
        logging.info('Creating dictionary (takes about 9h)..')

        # Construct corpus
        wiki = WikiCorpus(dump_name)

        # Remove words occuring less than 20 times, and words occuring in more
        # than 10% of the documents. (keep_n is the vocabulary size)
        wiki.dictionary.filter_extremes(no_below=10, no_above=0.1, keep_n=100000)

        # Save dictionary to file
        wiki.dictionary.save_as_text(dictionary_name)
        del wiki

    logger.info("loading dictionary")
    dictionary = Dictionary.load_from_text(dictionary_name)

    logger.info("loading corpus")
    wiki = WikiCorpus(dump_name, dictionary=dictionary)

    logger.info("getting sentences")
    sentences = list(wiki.get_texts())

    params = {'size': 400, 'window': 10, 'min_count': 10,
              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3}

    logger.info("training the model")
    word2vec = Word2Vec(sentences, **params)

    logger.info("storing the model")
    word2vec.save('bgwiki_word2vec')
