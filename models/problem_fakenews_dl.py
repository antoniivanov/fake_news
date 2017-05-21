import pandas as pd
import numpy as np
from keras.utils import np_utils
import logging
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.merge import maximum
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.optimizers import Adadelta

# preset seed for repeatability
seed = 7
np.random.seed(seed)

# main params
model_name = 'fakenews_model_799'
predict_mode = True

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
    data[column_name] = (data[column_name]) / (data[column_name].max() - data[column_name].min())
    return data

def load_model(modelfilename):
    model = None
    logger.info('loading model ' + modelfilename)
    with open(modelfilename + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)


    logger.info('loading weights')
    model.load_weights(modelfilename+ '.hdf5')
    return model

def get_model(name, shapes = [], iteration = 0):
    model = None
    if predict_mode:
        logger.info("loading the model")
        model = load_model('C:/Users/minimalistic/PycharmProjects/test/tmp/' + name)
    else:
        logger.info("building the model")
        model = build_model(shapes)

        logger.info('saving model to disk')
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)

    return model

def build_model(shapes):

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

    x1 = Dropout(0.9)(xmerged)
    xmerged = Dense(2, activation='softmax')(xmerged)

    model = Model([x2_input, x3_input], xmerged)

    return model

# setup logger
logger = setup_logger(model_name + "_training.log")

data = pd.read_csv('train_data_features.csv')
logger.info('processing label set')

# process the Ys (convert all 3s to 0, 1s stay 1)
data.loc[data['fake_news_score'] == 3, 'fake_news_score'] = 0
y_data = np_utils.to_categorical(data['fake_news_score'].as_matrix())

logger.info('processing x2: titles')
x2_data = np.load('./data/embedded_titles_400.npy')

logger.info('processing x3: content')
x3_data = np.load('./data/embedded_content_100.npy')
# trimming
x3_data = x3_data[:, :600, :]

logger.info('build model')
model = get_model(model_name, shapes=[y_data.shape, y_data.shape, x2_data.shape, x3_data.shape])


logger.info('compile model')
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(lr=0.5),
              metrics=['accuracy'])

if not predict_mode:
    # prep the validation split 0.2 for validation, 0.8 for train
    split = 0.2
    x2_train, x2_val, x3_train, x3_val, y_train, y_val = train_test_split(x2_data, x3_data, y_data, test_size=split, random_state=seed)

    # make sure we persist the model
    checkpointer = ModelCheckpoint(filepath="./tmp/fakenews_model.hdf5", verbose=1, save_best_only=False)

    model.fit([x2_train, x3_train], y_train,
              validation_data=([x2_val, x3_val], y_val),
              epochs=500,
              batch_size=32,
              callbacks=[checkpointer])
else:
    # get the test data
    data = pd.read_csv('test_data_features.csv')

    # load the embeddings for the test dataset
    title_data = np.load('./data/test_embedded_titles_400.npy')
    content_data = np.load('./data/test_embedded_content_100.npy')

    # trim the content to 600 words
    content_data = content_data[:, :600, :]

    # predict the test set
    predicted = model.predict([title_data, content_data], 32, verbose=1)

    # pick the selected category out of the softmax distribution
    classes = predicted.argmax(-1)

    # create the submission set
    res = pd.DataFrame();
    res['fake_news_score'] = classes
    res['fake_news_score'] = res['fake_news_score'].apply(pd.to_numeric)
    res['click_bait_score'] = data['click_bait_score'].copy()

    # replace 0s with 3 as this is how we encoded these for training
    res.loc[res['fake_news_score'] == 0, 'fake_news_score'] = 3

    res.to_csv('fake_news_submission_kolev.csv')

logger.info('Done')