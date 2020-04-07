'''
Concolutional Autoencoder 1.2
'''
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model, Sequential
from keras import backend as K
import scipy.io as sio
#load data
f1 = sio.loadmat('A:/Desktop/BA/data_sod/sod241Kn0p00001/f.mat')
f2 = sio.loadmat('A:/Desktop/BA/data_sod/sod25Kn0p00001/f.mat')
f3 = sio.loadmat('A:/Desktop/BA/data_sod/sod25Kn0p01/f.mat')

f1 = f1['f']
f2 = f2['f']
f3 = f3['f']

#Getting dimensions 
shape = f1.shape
t = shape[0] 
v = shape[1] 
x = shape[2] 
#dividing snapshots into different times for training
x_train = f1[:,:,:]
x_test = f2[:,:,:]
x_val = f3[:,:,:]

encoding_dim = 60
#------Encoder------
encoder_input = Input(shape=(40,200,1),name='original_data')
e = Conv2D(8,(5,5), activation='elu', padding='same', strides=(2,2))(encoder_input)
e = Conv2D(16,(5,5), activation='elu', padding='same', strides=(2,2))(e)
e = Conv2D(32,(5,5), activation='elu', padding='same', strides=(2,2))(e)
e = Conv2D(64,(5,5), padding='same', activation='elu', strides=(2,2))(e)
l = Flatten()(e)
encoder_output = Dense(encoding_dim, activation='elu')(l)

encoder = Model(encoder_input,encoder_output, name='encoder')
encoder.summary()
#-------Decoder------
decoder_input = Input(shape=(encoding_dim,), name='encoded_data')
l = Dense(125, activation = 'elu')(decoder_input)
d = Reshape((5,25,1))(l)
d = Conv2DTranspose(64,(5,5), activation='elu', padding= 'same', strides=(2,2))(d)
d = Conv2DTranspose(32,(5,5), activation = 'elu', padding= 'same', strides=(2,2))(d)
d = Conv2DTranspose(16,(5,5), activation='elu', padding='same')(d)
decoder_output = Conv2DTranspose(1,(5,5), activation='sigmoid', padding='same', strides=(2,2))(d)

decoder = Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

#------autoencoder------
autoencoder_input = Input(shape=(40,200,1), name ='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_percentage_error',metrics=["acc"])
autoencoder.summary()

#keras.utils.plot_model(autoencoder, 'autoencoder.png')

import numpy as np

x_train = (x_train - np.amin(x_train))/(np.amax(x_train)-np.amin(x_train))
x_test = (x_test - np.amin(x_test))/(np.amax(x_test)-np.amin(x_test))
x_val = (x_val - np.amin(x_val))/(np.amax(x_val)-np.amin(x_val))
x_train = np.reshape(x_train, (len(x_train),40,200,1))
x_test = np.reshape(x_test, (len(x_test),40,200,1))
x_val = np.reshape(x_train, (len(x_train),40,200,1))
x_val = np.reshape(x_test, (len(x_test),40,200,1))

history = autoencoder.fit(x_train,x_train,
                            epochs=400,
                            batch_size=128,
                            shuffle=True,
                            validation_split=0.25)
                

#-------evaluate the Model-----------------------
evaluate = autoencoder.evaluate(x_test,x_test, verbose=2)
print('Test loss:', evaluate[0])
print('Test accuracy:', evaluate[1])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

decoded_imgs = autoencoder.predict(x_val)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(40, 200))
    plt.colorbar()

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(40, 200))
    plt.colorbar()

plt.show()
