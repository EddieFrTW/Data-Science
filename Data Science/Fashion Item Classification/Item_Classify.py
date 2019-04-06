from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.advanced_activations import LeakyReLU


img_rows = 28
img_cols = 28
num_classes = 10
def generate_data(train_data, test_data):

    #Train data
    X = np.array(train_data.iloc[:, 1:])
    Y = to_categorical(np.array(train_data.iloc[:, 0]))
    X_test = np.array(test_data.iloc[:, 1:])
    Y_test = to_categorical(np.array(test_data.iloc[:, 0]))     
    
    datagen = ImageDataGenerator(
    zca_whitening=False,
    vertical_flip=False,
    horizontal_flip=True, fill_mode='nearest')

    aug_Img = [];
    aug_Label = [];
    ind = 0;
    for train_img in X:
        Img = train_img.reshape(1, img_rows, img_cols, 1)
        #Img = train_img
        Label = Y[ind]
        ind += 1;
        i = 0;
        for batch in datagen.flow(Img, batch_size=32):
            new_img = batch
            i = i+1;
            aug_Img.append(new_img)
            aug_Label.append(Label);
            if i > 2:
                break
    X = np.array(aug_Img).reshape(-1, img_rows*img_cols)
    Y = np.array(aug_Label)
    
    X = X.astype('float32')
    X_test = X_test.astype('float32')
    '''
    #Shuffle data
    index = np.arange(120000)
    np.random.shuffle(index)
    X = X[index,:]
    Y = Y[index]
    
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)    
    '''
    #split validation data, 20%    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=8)
    #Test data

    #reshape data
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255 
    X_val = X_val.astype('float32')/255
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def CNN_model():
    input_shape = (img_rows, img_cols, 1)
    
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation=None, padding = 'same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    
    model.add(Conv2D(32, (2, 2), activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2), activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (4, 4), activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(1024, activation=None))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
    
def ans_csvfile(pred):
    filename = '0660223_ans.csv'
    num_ans = len(pred)
    with open(filename, 'w+', newline='') as csv_file:  
        
        writer = csv.writer(csv_file)  
        writer.writerow(['id','label'])
        
        for i in range(num_ans): 
            ans = [str(i),str(pred[i])]
            writer.writerow(ans) 
    
if __name__ == '__main__':  

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    	    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = generate_data(train_data, test_data);

    
    model = CNN_model()
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    batch_size = 200
    epochs = 40
    # 進行訓練, 訓練過程會存在 history 變數中
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                        batch_size=batch_size, epochs=epochs, verbose=1)
    
    #score = model.evaluate(X_test, Y_test)
    #google-chrome
    
   # print('Test loss:', score[0])
   # print('Test accuracy:', score[1])
    #prediction
    predictions = model.predict_classes(X_test)
    ans_csvfile(predictions)

    



















