'''
Autoencoder No.1
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model
from keras.datasets import mnist

encoding_dim=32

#input placeholder
input_img = Input(shape=(784,))     
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
#"lossy" recontruction of input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
#-----------------------Encoder model-----------------------------------
#maps input to to encoded representation
encoder = Model(input_img, encoded)


#--------Decoder model--------------------------------------------------
#placeholder for encoded input 32 dimensional
encoded_input = Input(shape=(32,))

#retrieve last layer of autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
#create decoder model
decoder = Model(input = encoded_input, output = decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))


#----------------------------------------------------------------------
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#load data
(x_train,_), (x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test,x_test))
                


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i +1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

