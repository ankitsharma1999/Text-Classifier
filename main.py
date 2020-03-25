import re
import sys
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
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding, Input, Bidirectional, GRU

class Text_Classifier():

    def __init__(self, path_to_data, path_to_emb, emb_size, lstm1Units, lstm2Units, d_1, d_2):
        self.path_to_data = path_to_data
        self.emb_size = emb_size
        self.path_to_emb = path_to_emb
        self.lstm1Units = lstm1Units
        self.lstm2Units = lstm2Units
        self.d_1 = d_1
        self.d_2 = d_2
    
    def loadData(self):
        df = pd.read_csv(self.path_to_data)
        X_train, y_train = np.asarray(df['text']), np.asarray(df['target'])
        return X_train, y_train

    def preprocessing(self, X_train):
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

    def loadEmbeddings(self, X_train, tok, v_size):
        embedding_vector = {}
        print('Loading Glove...')
        f = open(self.path_to_emb)
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef

        embedding_matrix = np.zeros((v_size,self.emb_size))
        for word,i in tqdm(tok.word_index.items()):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value
        print('Glove Loaded.')
        return embedding_matrix
    
    def modelArchitecture_1(self,v_size, maxLen, emb_matrix):
        ip = Input(shape=(maxLen,))
        emb = Embedding(v_size,self.emb_size,weights = [emb_matrix],trainable = False)(ip)
        lstm_1 = Bidirectional(LSTM(self.lstm1Units, return_sequences=True))(emb)
        drop_1 = Dropout(self.d_1)(lstm_1)
        lstm_2 = LSTM(self.lstm2Units)(drop_1)
        drop_2 = Dropout(self.d_2)(lstm_2)
        dense_1 = Dense(1)(drop_2)
        op = Activation('sigmoid')(dense_1)
        model = Model(inputs=ip, outputs=op)
        print(model.summary())
        return model

    def modelArchitecture_2(self,v_size, maxLen, emb_matrix):
        ip = Input(shape=(maxLen,))
        emb = Embedding(v_size,self.emb_size,weights = [emb_matrix],trainable = False)(ip)
        lstm_1 = Bidirectional(GRU(self.lstm1Units, return_sequences=True))(emb)
        drop_1 = Dropout(self.d_1)(lstm_1)
        lstm_2 = GRU(self.lstm2Units)(drop_1)
        drop_2 = Dropout(self.d_2)(lstm_2)
        dense_1 = Dense(1)(drop_2)
        op = Activation('sigmoid')(dense_1)
        model = Model(inputs=ip, outputs=op)
        print(model.summary())
        return model
    
    def plot_metrics(self, history):
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.plot()
        plt.show()

        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.plot()
        plt.show()

    def train(self, epochs=20, batch_size=28, validation_split=0.1, shuffle=True, opt='adam', plot_modelArchitecture=False, plot_diagnostics=True, option='lstm'):
        X_train, y_train = self.loadData()
        X_train, maxLen, v_size, tok = self.preprocessing(X_train)
        emb_matrix = self.loadEmbeddings(X_train, tok, v_size)
        if option == 'lstm':
            model = self.modelArchitecture_1(v_size, maxLen, emb_matrix)
        elif option == 'gru':
            model = self.modelArchitecture_2(v_size, maxLen, emb_matrix)

        if plot_modelArchitecture:
            plot_model(model, to_file='model_final_{}.png'.format(int(time.time())))
        
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        print("Training....")
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, shuffle=shuffle)

        if plot_diagnostics:
            self.plot_metrics(history)
        
        return model, history

    def predict(self, X_test, model):
        X_test, _, _, _ = self.preprocessing(X_test)
        res = model.predict(X_test)
        return res

if __name__ == '__main__':
    data = 'Tweets/train.csv'
    embd = 'Glove/glove.6B.50d.txt'
    clf = Text_Classifier(data, embd, 50, 10, 4, 0.5, 0.5)
    arg = (sys.argv)[1]
    model, history = clf.train(option=arg)
    df = pd.read_csv('Tweets/test.csv')
    X_test = np.asarray(df['text'])
    res = clf.predict(X_test, model)
