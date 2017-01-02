import iop1 as io
import sys
import numpy as np

from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization


def uniquifyData(X):
    return (np.unique(X[:,0]) , np.unique(X[:,1]), np.unique(X[:,2]))

def oneHotFeature(X, fList):
    X0 = (X[:,0][:,None] == fList[0]).astype('float')
    X1 = (X[:,1][:,None] == fList[1]).astype('float')
    X2 = (X[:,2][:,None] == fList[2]).astype('float')
    #X0 = predict2class(X0).reshape([X0.shape[0],1])
    #X11 = predict2class(X1[:, 0:15]).reshape([X1.shape[0],1])
    #X12 = predict2class(X1[:,15:30]).reshape([X1.shape[0],1])
    #X13 = predict2class(X1[:,30:45]).reshape([X1.shape[0],1])
    #X14 = predict2class(X1[:,45:60]).reshape([X1.shape[0],1])
    #X15 = predict2class(X1[:,60:  ]).reshape([X1.shape[0],1])
    #X1  = np.concatenate( (X11,X12,X13,X14,X15), axis=1 )
    #X1 = predict2class(X1).reshape([X1.shape[0],1])
    #X2 = predict2class(X2).reshape([X2.shape[0],1])
    return (X0,X1,X2)

def normalize( X, Y ):
    return (X-Y.mean(axis=0)) / Y.std(axis=0)

def combine( hot, norm ):
    return np.concatenate( (hot[0],hot[1],hot[2],norm), axis=1 )


def shuffle( X, Y ):
    vNum = 3600000
    tmp = np.concatenate((X,Y), axis = 1)
    np.random.shuffle(tmp)
    newX = tmp[:, 0:-42]
    newY = tmp[:, -42:]
    
    X_t = newX[0:vNum]
    X_v = newX[vNum:]
    Y_t = newY[0:vNum]
    Y_v = newY[vNum:]
    return X_t, Y_t, X_v, Y_v 

def createModel_DNN():
    model = Sequential()
    
    # fully connected ntk
    model.add( Dense(input_dim=121, output_dim=80) )
    model.add( BatchNormalization() )
    model.add( Activation('relu') )
    model.add( Dropout(0.5) )
    
    model.add( Dense(60) )
    model.add( BatchNormalization() )
    model.add( Activation('relu') )
    model.add( Dropout(0.5) )
    
    model.add( Dense(42) )
    model.add( Activation('softmax') )
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def predict2class(Yp):
    return np.argmax(Yp, axis=1)


if __name__ == '__main__':
    dataPath  = './data/train'    
    classPath = './data/training_attack_types.txt' 
    testPath  = './data/test.in'
    outPath   = './myout_ba32.csv'
    X_word, X_num, Y = io.loadData( dataPath )
    print '>> preprocessing X'
    uFeature  = uniquifyData(X_word)
    X_hot  = oneHotFeature(X_word, uFeature)
    X_all  = combine(X_hot, X_num)
    #X_norm = normalize(X_num, X_num)
    X_train = normalize(X_all, X_all)
    
    print '>> preprocessing Y'
    d = io.loadAttack( classPath )
    Y_label = np.array([ d[Y[x]] for x in range(Y.shape[0])] )
    Y_train = np_utils.to_categorical( Y_label, 42 )
     
    print '>> creating DNN and start to train'
    model0 = createModel_DNN()
    

    
    epoch = 150
    X_t, Y_t, X_v, Y_v = shuffle(X_train, Y_train)
    model0.fit( x=X_t, y=Y_t, batch_size=32, nb_epoch=epoch, validation_data=(X_v,Y_v) )
    
    #model.fit( x=X_t,  y=Y_t, batch_size=100, nb_epoch=10)

    print '>> loading testData.'
    T_word, T_num = io.loadTest(testPath)
    T_hot  = oneHotFeature(T_word, uFeature)
    T_all  = combine(T_hot, T_num)
    X_test = T_all
    for ii in range(121):
        X_test[:,ii] = (T_all[:,ii] - X_all[:,ii].mean())/X_all[:,ii].std()

    print '>> predicting Y'
    Y_test = model0.predict(X_test)
    
    T_class = predict2class(Y_test)
    io.attack2class(T_class)
    io.outputCSV( outPath, T_class )
