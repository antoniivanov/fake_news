import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
     prog='feature_extract',
     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_data', type=str, default="./data/FN_Training_Set.csv", help='Path to original training set')

args = parser.parse_args()

print(args.train_data)

train_data = pd.read_csv(args.train_data, encoding='windows-1251')

with open('bg_stopwords.txt', encoding='utf8') as f:
    stopwords = [x.strip() for x in f.readlines()]
stopwords

def sentence2wordlist(raw, language='bg+en'):
    """language: 'bg+en', 'bg', 'en', 'symbol'"""
    if language == 'bg+en':
        regex = "[^а-яА-Яa-zA-Z]"
    elif language == 'bg':
        regex = "[^а-яА-Я]"
    elif language == 'en':
        regex = "[^a-zA-Z]"
    elif language == 'symbol':
        regex = "[^-!$%^&*()_+|~=`{}\[\]:\";'<>?,.\/]"
    elif language == '!':
        regex = "[^?!]"
    clean = re.sub(regex," ", raw)
    words = clean.split()
    return words

get_number_words = lambda sent: len(sentence2wordlist(sent))

get_number_char = lambda sent: len(sent)

get_number_symbols = lambda sent: len(sentence2wordlist(sent, 'symbol'))

def get_number_stopwrods(sent):
    wordlist = sentence2wordlist(sent)
    return array(list(map(lambda x: x in stopwords, wordlist))).sum()
    
def get_avg_char_per_word(sent):
    wordlist = sentence2wordlist(sent)
    return array(list(map(len, wordlist))).mean()

def get_avg_caps_per_char(sent):
    chars_re = "[^а-яА-Яa-zA-Z]"
    # remove white spaces as well as symbols
    clean = re.sub(chars_re,"", sent)
    caps_re = "[^А-ЯA-Z]"
    caps = re.sub(caps_re, "", clean)
    try:
        return len(caps)/len(clean)
    except:
        return -1 # div by 0 case

func_list = [get_number_words, 
             get_number_char,
             get_number_symbols,
             get_number_stopwrods,
             get_avg_char_per_word,
             get_avg_caps_per_char]

col_name = lambda s: list(map(lambda x: x.format(ph=s), col_name_ph))

col_name_ph = ["{ph}_number_words", 
             "{ph}_number_char",
             "{ph}_number_symbols",
             "{ph}_number_stopwords",
             "{ph}_avg_char_per_word",
             "{ph}_avg_caps_per_char"]

def append_new_columns(df, column, name_func_dict):
    for col, func in name_func_dict.items():
        print("Adding col: {}".format(col))
        df[col] = df[column].apply(func)
    return 
