# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from text_cleaner import TextCleaner
import logging

log = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.DEBUG)


def start(args):
    df = pd.read_csv(args.input_file, encoding='utf-8')
    if (log.isEnabledFor(logging.DEBUG)):
        log.debug(df.head(10))

    if (args.limit > 0):
        log.warn("Limit rows to %d", args.limit)
        df = df.head(args.limit)

    cleaner = TextCleaner()
    df = cleaner.clean(df)


if __name__ == '__main__':
    # iconv -f 'windows-1251' -t 'utf-8' FN_Training_Set.csv > FN_Training_Set.utf8.csv

    parser = argparse.ArgumentParser(description='Start fake news')
    parser.add_argument(
        '--input_file', type=str, help='input csv file', default='../data/FN_Training_Set.utf8.csv')
    parser.add_argument(
        '--limit', type=long, help='input csv file', default=10)
    args = parser.parse_args()

    log.info("Start with arguments: %s", args)

    start(args)

    pass
