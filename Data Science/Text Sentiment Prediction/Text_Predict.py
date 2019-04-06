import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import csv
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model    
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Embedding, LSTM
from keras.layers import GlobalAveragePooling1D, Conv1D, MaxPooling1D, Bidirectional, TimeDistributed, Concatenate
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
MAX_num_words = 20000
embed_dim = 300
lstm_out = 128
batch_size = 200
epochs = 5
def preprocessing_data(_data):
    
    #preprocessing training data
    _data['text'] = _data['text'].apply(lambda x: x.lower())

    _data['text'] = _data['text'].apply((lambda x: re.sub(r'@[\S]+', '', x)))
     # Replaces #hashtag with hashtag
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'#(\S+)', r' \1 ', x)))
    #punctuation HTML type
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'&[\S]+', '', x)))
    #URL
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', x)))
    #Remove not alphanumeric symbols white spaces
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'[^\w]', ' ', x)))
    
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'\brt\b', '', x)))
    #repeating characters
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(.)\1{1,}', r'\1\1', x)))
    #Emotion
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', x)))
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D    
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', x)))
    # Love -- <3, :*
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(<3|:\*)', ' EMO_POS ', x)))
    # Sad -- :-(, : (, :(, ):, )-:
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', x)))
    # Cry -- :,(, :'(, :"(
    _data['text'] = _data['text'].apply((lambda x: re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', x)))

    _data['text'] = _data['text'].apply((lambda x: re.sub(r'\s+', r' ', x)))

    '''
    #load stop word list
    text_file = open("twitter-stopwords.txt", "r")
    STOP_WORDS_list = text_file.read().split(',')
    '''
    
    return _data

def generate_data(train_dataX, train_dataY, test_data):

    # Keeping only the neccessary columns
    tokenizer = Tokenizer(num_words = MAX_num_words, split=' ')
    #train data
    tokenizer.fit_on_texts(train_dataX['text'].values)
    X = tokenizer.texts_to_sequences(train_dataX['text'].values)
        
    X = pad_sequences(X, maxlen=100, padding='post')
    
    '''    
    Y = train_data['sentiment'].values
    num_train = len(Y)
    for i in range(num_train):
        if Y[i] == 4:
            Y[i] = 1
    '''
    #one-hot encoding
    Y = pd.get_dummies(train_dataY['sentiment']).values
    
    #Shuffle data
    index = np.arange(len(X))
    np.random.shuffle(index)
    X = X[index,:]
    Y = Y[index]
    
    #split validation data, 20%    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=8)
    
    #test data
    X_test = tokenizer.texts_to_sequences(test_data['text'].values)
    X_test = pad_sequences(X_test, maxlen=100, padding='post')
   
    return tokenizer, X_train, Y_train, X_val, Y_val, X_test

def pre_train_embedding_matrix(tokenizer, embeddings_index):
    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = min(MAX_num_words, len(word_index))
    embedding_matrix = np.zeros((num_words, embed_dim))
    
    for word, i in word_index.items():
        if i >= MAX_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def LSTM_CNN_model(X_train, embedding_matrix):
        
    #embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
    #                        input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    #CNN-LSTM
    '''
    model1 = Sequential()
    model1.add(Embedding(MAX_num_words, embed_dim, weights=[embedding_matrix], input_length = X_train.shape[1]))
    model1.add(Dropout(0.2))
    model1.add(Conv1D(64, kernel_size=3, padding='same', activation=None))
    model1.add(BatchNormalization())
    model1.add(LeakyReLU(alpha=0.1)) 
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Dropout(0.1))       
    model1.add(Bidirectional( LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True) ))
    model1.add(TimeDistributed(Dense(1, activation='tanh'))) 
    model1.add(Flatten())
    model1.add(Dense(2, activation='softmax'))
    '''
    #LSTM-CNN
    model = Sequential()
    model.add(Embedding(MAX_num_words, embed_dim, input_length = X_train.shape[1]))
    model.add(Dropout(0.1))    
    model.add(Bidirectional( LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True) ))
    model.add(Conv1D(32, kernel_size=2, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1)) 
    model.add(Flatten())
    model.add(Dense(128, activation=None))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    #print(model.summary())
    return model

def CNN_LSTM_model(X_train, embedding_matrix):
        #CNN-LSTM
    
    model = Sequential()
    model.add(Embedding(MAX_num_words, embed_dim, input_length = X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1)) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))       
    model.add(Bidirectional( LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True) ))
    model.add(TimeDistributed(Dense(1, activation='tanh'))) 
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model
    

def CNN_model(X_train, embedding_matrix):
    #CNN
    modela = Sequential()
    modela.add(Embedding(MAX_num_words, embed_dim, input_length = X_train.shape[1]))
    #model1.add(Dropout(0.2))
    modela.add(Conv1D(32, kernel_size = 3, padding = 'same', activation=None))
    modela.add(BatchNormalization())
    modela.add(LeakyReLU(alpha=0.1))
    modela.add(MaxPooling1D(pool_size=2))
    modela.add(Dropout(0.1))
 
    modela.add(Conv1D(32, kernel_size = 3, padding = 'same', activation=None))
    modela.add(BatchNormalization())
    modela.add(LeakyReLU(alpha=0.1))
    modela.add(MaxPooling1D(pool_size=2))
    modela.add(Dropout(0.1))    
    
    modela.add(Conv1D(64, kernel_size = 3, padding = 'same', activation=None))
    modela.add(BatchNormalization())
    modela.add(LeakyReLU(alpha=0.1))
    modela.add(MaxPooling1D(pool_size=2))
    modela.add(Dropout(0.1))
    #modela.add(Flatten())
    modela.add(GlobalAveragePooling1D())

    modelb = Sequential()
    modelb.add(Embedding(MAX_num_words, embed_dim, input_length = X_train.shape[1]))
    #model1.add(Dropout(0.2))
    modelb.add(Conv1D(32, kernel_size = 4, padding = 'same', activation=None))
    modelb.add(BatchNormalization())
    modelb.add(LeakyReLU(alpha=0.1))
    modelb.add(MaxPooling1D(pool_size=2))
    modelb.add(Dropout(0.1))

    modelb.add(Conv1D(32, kernel_size = 4, padding = 'same', activation=None))
    modelb.add(BatchNormalization())
    modelb.add(LeakyReLU(alpha=0.1))
    modelb.add(MaxPooling1D(pool_size=2))
    modelb.add(Dropout(0.1))
    
    modelb.add(Conv1D(64, kernel_size = 4, padding = 'same', activation=None))
    modelb.add(BatchNormalization())
    modelb.add(LeakyReLU(alpha=0.1))
    modelb.add(MaxPooling1D(pool_size=2))
    modelb.add(Dropout(0.1))
    #modelb.add(Flatten())
    modelb.add(GlobalAveragePooling1D())

    conc = Concatenate()([modela.output, modelb.output])
    
    #notice you concatenate outputs, which are tensors.     
    #you cannot concatenate models
    
    out = LeakyReLU(alpha=0.1)(conc)
    out = Dropout(0.5)(out)
    out = Dense(2, activation='softmax')(out)
    
    model = Model([modela.input, modelb.input], out)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model
'''
def CNN_mdeol2(X_train, embedding_matrix):
    
    model = Sequential()
    model.add(Embedding(MAX_num_words, embed_dim, input_length = X_train.shape[1]))
    model.add(Conv1D(32, kernel_size = 3, padding = 'same', activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
 
    model.add(Conv1D(32, kernel_size = 3, padding = 'same', activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))    
    
    model.add(Conv1D(64, kernel_size = 3, padding = 'same', activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))    
    return model
'''
def ans_csvfile(predictions):
    
    #using one-hot encoding
    filename = '0660223_ans.csv'
    num_ans = len(predictions)
    with open(filename, 'w+', newline='') as csv_file:      
        writer = csv.writer(csv_file)  
        writer.writerow(['id','Sentiment'])
        
        for i in range(num_ans):
            ans = [str(i),str(predictions[i][1]*4)]
            writer.writerow(ans)
    '''
    filename = '0660223_ans.csv'
    num_ans = len(predictions)
    with open(filename, 'w+', newline='') as csv_file:      
        writer = csv.writer(csv_file)  
        writer.writerow(['id','Sentiment'])
        
        for i in range(num_ans):
            ans = [str(i),str(float(predictions[i]*4))]
            writer.writerow(ans)
    '''
    #%%
if __name__ == '__main__':  
    global MAX_num_words
    num_words = [20000]
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    train_dataX = train_data[['text']]
    train_dataY = train_data[['sentiment']]
    test_data = test_data[['text']]
    
    
    train_dataX = preprocessing_data(train_dataX)
    test_data = preprocessing_data(test_data)
    
    
    #load pre-trained word vector from "glove.6B.100d.txt"
    '''
    embeddings_index = {}
    f=open("glove.6B.300d.txt", 'r', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    '''
    #%%
    for i in range(len(num_words)):
        MAX_num_words = num_words[i]
        tokenizer, X_train, Y_train, X_val, Y_val, X_test = generate_data(train_dataX, train_dataY, test_data)
        
        embedding_matrix = pre_train_embedding_matrix(tokenizer, embeddings_index)
        model1 = LSTM_CNN_model(X_train, embedding_matrix)
        model2 = CNN_LSTM_model(X_train, embedding_matrix)
        model3 = CNN_model(X_train, embedding_matrix)
        # 進行訓練
        
        model1.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                  batch_size=batch_size, epochs=epochs, verbose=1)    
        model2.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                  batch_size=batch_size, epochs=epochs, verbose=1)
        
        model3.fit([X_train, X_train], Y_train, validation_data=([X_val, X_val], Y_val), 
                  batch_size=batch_size, epochs=3, verbose=1)

    predict1 = model1.predict(X_test)
    predict2 = model2.predict(X_test)
    predict3 = model3.predict([X_test, X_test])
    predictions = predict1*0.3334 + predict2*0.3333 + predict3*0.3333
    ans_csvfile(predictions)
