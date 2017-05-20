import pdb
import gensim
import pandas as pd
import numpy as np
from keras.utils import np_utils
import logging


from keras.regularizers import l1, l2

from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Embedding, Dropout, Activation
from keras.layers.merge import concatenate, Concatenate, maximum
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.optimizers import Adam, Adadelta
from keras import losses

seed = 7
np.random.seed(seed)

# dimensionality of w2v embeddings
word2vec_dim = 100

# some constants
# TODO: should be read at runtime once we switch to the real CSV
max_body_len = 5337
max_title_len = 41

model_name = 'x1_only'
predict_mode = False


def setup_logger(filename):
    logger = logging.getLogger('scope.name')
    logger.setLevel(logging.DEBUG)

    file_log_handler = logging.FileHandler(filename)
    logger.addHandler(file_log_handler)

    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)

    # nice output format
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    return logger

def normalize_column(data, column_name):
    data[column_name] = (data[column_name] - data[column_name].mean()) / (data[column_name].max() - data[column_name].min())
    return data

def load_model(modelfilename, iteration):
    model = None
    logger.info('loading model ' + modelfilename)
    with open(modelfilename + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

    logger.info('loading weights')
    model.load_weights(modelfilename + '_iteration' + str(iteration) + '.h5')
    return model

def get_model(name, shapes = [], iteration = 0):
    model = None
    if predict_mode:
        logger.info("loading the model")
        model = load_model('.\\model_state\\' + name, iteration)
    else:
        logger.info("building the model")
        model = build_model(shapes)

        logger.info('saving model to disk')
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)

    return model

def build_model(shapes):

    '''
    x1_input = Input(shape=(shapes[1][1],))
    x1 = Dense(32, activation='tanh')(x1_input)
    '''

    x2_input = Input(shape=(shapes[2][1], shapes[2][2]))
    x2 = Conv1D(32, 5, activation='relu')(x2_input)
    x2 = Conv1D(32, 10, activation='relu')(x2)
    x2 = MaxPooling1D(5)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(32, activation='relu')(x2)

    x3_input = Input(shape=(shapes[3][1], shapes[3][2]))
    x3 = Conv1D(128, 20, activation='relu')(x3_input)
    x3 = MaxPooling1D(5)(x3)
    x3 = Conv1D(256, 20, activation='relu')(x3)
    x3 = MaxPooling1D(5)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(32, activation='relu')(x3)

    xmerged = maximum([x2, x3])
    #xmerged = x3

    #xmerged = Dense(32, activation='tanh')(xmerged)
    x1 = Dropout(0.8)(xmerged)
    xmerged = Dense(2, activation='softmax')(xmerged)

    model = Model([x2_input, x3_input], xmerged)

    return model

# setup logger
logger = setup_logger(model_name + "_training.log")

data = pd.read_csv('train_data_features.csv')
logger.info('processing label set')

data.loc[data['fake_news_score'] == 3, 'fake_news_score'] = 0
y_data = np_utils.to_categorical(data['fake_news_score'].as_matrix())

columns_to_normalize = ['body_number_words',
                'body_number_symbols',
                'body_number_stopwords',
                'title_number_words',
                'title_number_symbols',
                'title_number_stopwords']

x1_all_columns = ['click_bait_score',
                'title_avg_char_per_word',
                'title_avg_caps_per_char',
                'body_avg_char_per_word',
                'body_avg_caps_per_char'] + columns_to_normalize


logger.info('processing x1: features')
x1_data = data[x1_all_columns].copy()

for column in columns_to_normalize:
    x1_data = normalize_column(x1_data, column)


logger.info('processing x2: titles')
x2_data = np.load('./data/embedded_titles_100.npy')

logger.info('processing x3: content')
x3_data = np.load('./data/embedded_content_100.npy')
# trimming
x3_data = x3_data[:, :600, :]

logger.info('build model')
model = get_model(model_name, shapes=[y_data.shape, x1_data.shape, x2_data.shape, x3_data.shape])

model.summary()

logger.info('compile model')
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(lr=0.2),
              metrics=['accuracy'])

# prep the validation split
x1_train, x1_val, y_train, y_val = train_test_split(x1_data.as_matrix(), y_data, test_size=0.1, random_state=seed)
x2_train, x2_val, _, _ = train_test_split(x2_data, y_data, test_size=0.1, random_state=seed)
x3_train, x3_val, _, _ = train_test_split(x3_data, y_data, test_size=0.1, random_state=seed)

model.fit([x2_train, x3_train], y_train,
          validation_data=([x2_val, x3_val], y_val),
          epochs=500,
          batch_size=128)

logger.info('Done')

