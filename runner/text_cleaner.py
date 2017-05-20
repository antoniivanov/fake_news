# -*- coding: utf-8 -*-

from StringIO import StringIO
import logging
import re
import io



log = logging.getLogger(__name__)

def read_lines(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        return [unicode(x.strip()) for x in f.readlines()]


class TextCleaner():

    def __init__(self):
        self.rules = {
            u'<EMAIL>': ur"\b[a-zа-я0-9._%+-]+@[а-яa-z0-9.-]+\.[а-яa-z]{2,}\b",
        }
        self.stopWords = read_lines('../feature_extraction/bg_stopwords.txt')


    def clean(self, df):
        log.info("Apply cleaner to title")
        df['Content Title'] = df['Content Title'].apply(lambda t: self.cleanText(t))

        if (log.isEnabledFor(logging.DEBUG)):
            log.debug(df['Content Title'].head(10))

    def replace_stopwords (self, s):
        # TODO: use nltk word_tokenizer or any kind of python tokeneize that works with unicode ??
        words = []
        for word in s.split(' '):
            if (word.strip() in self.stopWords):
                word = u'<STOPWORD>'

            words.append(word)
        return ' '.join(words)

    def cleanText(self, text):

        text = text.strip().replace(u"\n", u" ").replace(u"\r", u" ")  # remove new lines
        text = text.lower()

        text = self.replace_stopwords(text)

        for name, rule_regex in self.rules.iteritems():
            ruleFinder = re.compile(rule_regex, re.IGNORECASE | re.UNICODE)
            text = ruleFinder.sub(name, text)

        return text

if __name__ == '__main__':
    pass
