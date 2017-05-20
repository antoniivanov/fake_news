
# coding: utf-8

# # NB
# 
# If you want to re-run this localy you will need to fix some of the file paths as the full data is not uploaded in the github repo.

# In[25]:

from numpy import *
import pandas as pd

import argparse
parser = argparse.ArgumentParser(
     prog='feature_extract',
     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_data', type=str, default="../data/FN_Training_Set.csv", help='Path to original training set')
parser.add_argument('--main_data_fake_news', type=str, default="../data/main_data_fake_news.csv", help='Path to original main data set')
parser.add_argument('--output_path', type=str, default="../data/train_data_features.csv", help='Path where to save csv with new features')

args = parser.parse_args()

# # Main (unlabled) data

# In[222]:

main_data_fake_news = pd.read_csv(args.main_data_fake_news) # Not here in the repo


# In[3]:

main_data_fake_news.head()


# In[38]:

main_data_fake_news[main_data_fake_news.title.isnull()]


# In[6]:

main_data_fake_news[main_data_fake_news.content.isnull()]


# # Train (labeled) data

# In[19]:

train_data = pd.read_csv(args.train_data, encoding='windows-1251')
train_data.head()


# ## Check for nulls

# In[22]:

train_data.info()


# In[106]:

train_data[train_data["Content Published Time"].isnull()]


# In[20]:

train_data[train_data["Content Title"].isnull()]


# In[21]:

train_data[train_data["Content"].isnull()]


# In[109]:

train_data[train_data["Content Url"].isnull()]


# In[107]:

train_data[train_data["fake_news_score"].isnull()]


# In[108]:

train_data[train_data["click_bait_score"].isnull()]


# In[111]:

train_data.fillna(value="", inplace=True)


# # Explore data distirbution

# In[23]:

train_data.describe()


# In[26]:

train_data.fake_news_score.plot.hist()


# In[27]:

train_data.click_bait_score.plot.hist()


# In[28]:

train_data[train_data.click_bait_score != train_data.fake_news_score]


# In[48]:

title_gb = train_data.groupby('Content Title')
dublicates = title_gb.aggregate({'Content Title':'count',
                    'fake_news_score':'mean',
                    'click_bait_score':'mean',
                   'Content Url': lambda x: x.nunique(),
                    'Content': lambda x: x.nunique(),
                   'Content Published Time': lambda x: " | ".join(x)
                   })


# In[91]:

dublicates[dublicates["Content Title"] > 1].sort_values('Content Title', ascending=False)


# #  Feature building
# ## Timestamp, hour, minute

# In[92]:

import datetime
str2datetime = lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M")
train_data['published_ts'] = train_data['Content Published Time'].apply(str2datetime)


# In[117]:

train_data['published_hour'] = train_data.published_ts.dt.hour
train_data['published_minute'] = train_data.published_ts.dt.minute


# In[99]:

train_data.published_ts.dt.hour.plot.hist()


# In[100]:

train_data.published_ts.dt.minute.plot.hist()


# ## stopwords

# In[217]:

import re
import pandas as pd

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
    return df


# In[218]:

train_data_extra_features = append_new_columns(train_data, 
                                                   "Content Title", 
                                                   dict(zip(col_name('title'),func_list)))

train_data_extra_features = append_new_columns(train_data_extra_features, 
                                                   "Content", 
                                                   dict(zip(col_name('body'),func_list)))


# In[221]:

print("Saving at {}".format(args.output_path))
train_data_extra_features.to_csv(args.output_path)


# In[ ]:



