import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding, Input, Bidirectional

df = pd.read_csv('Tweets/train.csv')
X_train, y_train = np.asarray(df['text']), np.asarray(df['target'])

#maxLen = len(max(X_train, key=len).split())

def preprocess(X_train):
    for i in range(len(X_train)):
        X_train[i] = X_train[i].lower()
        X_train[i] = re.sub(r'\d+', '', X_train[i])
        X_train[i] = re.sub(r'[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', '', X_train[i])
        X_train[i].strip()

    tok = Tokenizer()
    tok.fit_on_texts(X_train)
    X_train = tok.texts_to_sequences(X_train)
    maxLen = len(max(X_train, key=len))
    X_train = pad_sequences(X_train, maxlen=maxLen)
    v_size = len(tok.word_index) + 1
    return X_train, maxLen, v_size, tok

X_train, maxLen, v_size, tok = preprocess(X_train)

embedding_vector = {}
f = open('Glove/glove.6B.50d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef

embedding_matrix = np.zeros((v_size,50))
for word,i in tqdm(tok.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

print('GloVe Loaded')

def model_1(v_size, maxLen, emb_matrix, glove_dim, d_1, d_2):
    ip = Input(shape=(maxLen,))
    emb = Embedding(v_size,glove_dim,weights = [emb_matrix],trainable = False)(ip)
    lstm_1 = Bidirectional(LSTM(10, return_sequences=True))(emb)
    drop_1 = Dropout(d_1)(lstm_1)
    lstm_2 = LSTM(4)(drop_1)
    drop_2 = Dropout(d_2)(lstm_2)
    dense_1 = Dense(1)(drop_2)
    op = Activation('sigmoid')(dense_1)
    model = Model(inputs=ip, outputs=op)
    print(model.summary())
    return model

model = model_1(v_size, maxLen, embedding_matrix, 50, d_1=0.5, d_2=0.5)
plot_model(model, to_file='model_final_{}.png'.format(int(time.time())))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training....")
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=28, shuffle=True)

def plot_loss(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.plot()
    plt.show()

def plot_accuracy(history):
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.plot()
    plt.show()
