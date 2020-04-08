'''
Autoencoder 1.2
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io as sio
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model


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

#Data Normalization as Proposed by Section 6.2 Sandia
x_train = (x_train - np.amin(x_train))/(np.amax(x_train)-np.amin(x_train))
x_test = (x_test - np.amin(x_test))/(np.amax(x_test)-np.amin(x_test))
x_val = (x_val - np.amin(x_val))/(np.amax(x_val)-np.amin(x_val))
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

encoding_dim=32

#input placeholder
input_img = Input(shape=(x*v,))     
print(input_img.shape)
encoded = Dense(128, activation='relu')(input_img)
print(encoded.shape)
encoded = Dense(64, activation='relu')(encoded)
print(encoded.shape)
encoded = Dense(encoding_dim, activation='relu')(encoded)
(encoded.shape)
#"lossy" recontruction of input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(v*x, activation='relu')(decoded)

autoencoder = Model(input_img, decoded)
#-----------------------Encoder model-----------------------------------
#maps input to to encoded representation
encoder = Model(input_img, encoded)


#--------Decoder model--------------------------------------------------
#placeholder for encoded input 32 dimensional
encoded_input = Input(shape=(encoding_dim,))

#retrieve last layer of autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
#create decoder model
decoder = Model(input = encoded_input, output = decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))


#----------------------------------------------------------------------
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_percentage_error',metrics=['acc'])


history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=20,
                shuffle=True,
                validation_data=(x_test,x_test))
                
                
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





encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
sio.savemat('A:/Desktop/BA/data_sod/sod25Kn0p00001auto/f.mat', mdict={'fa': decoded_imgs})  
              
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)
sio.savemat('A:/Desktop/BA/data_sod/sod241Kn0p00001auto/f.mat', mdict={'fa': decoded_imgs})

encoded_imgs = encoder.predict(x_val)
decoded_imgs = decoder.predict(encoded_imgs)
sio.savemat('A:/Desktop/BA/data_sod/sod25Kn0p01auto/f.mat', mdict={'fa': decoded_imgs})





n=2
plt.figure(figsize=(20,2))
for i in range(n):
    
    ax = plt.subplot(2,n,i +1)
    plt.imshow(x_test[i+20].reshape(v,x))
    plt.gray()
    ax = plt.subplot(2,n,i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(v,x))
    plt.gray()
plt.show()

