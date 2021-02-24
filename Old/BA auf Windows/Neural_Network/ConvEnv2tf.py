'''
Concolutional Autoencoder 1.2
'''
import matplotlib.pyplot as plt
import tensorflow as tf
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
#dividing snapshots
x_train = f1
x_test = f2
x_val = f3

import numpy as np

x_train = (x_train - np.amin(x_train))/(np.amax(x_train)-np.amin(x_train))
x_test = (x_test - np.amin(x_test))/(np.amax(x_test)-np.amin(x_test))
x_val = (x_val - np.amin(x_val))/(np.amax(x_val)-np.amin(x_val))
x_train = np.reshape(x_train, (len(x_train),40,200,1))
x_test = np.reshape(x_test, (len(x_test),40,200,1))
x_val = np.reshape(x_train, (len(x_train),40,200,1))
x_val = np.reshape(x_test, (len(x_test),40,200,1))

latent_dim = 10


encoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(40, 200, 1)),
        tf.keras.layers.Conv2D(8,(5,5), activation='elu', padding='same', strides=(2,2)),
        tf.keras.layers.Conv2D(16,(5,5), activation='elu', padding='same', strides=(2,2)),
        tf.keras.layers.Conv2D(32,(5,5), activation='elu', padding='same', strides=(2,2)),
        tf.keras.layers.Conv2D(64,(5,5), padding='same', activation='elu', strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim, activation='elu'),
        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        tf.keras.layers.Dense(125, activation = 'elu'),
        tf.keras.layers.Reshape((5,25,1)),
        tf.keras.layers.Conv2DTranspose(64,(5,5), activation='elu', padding= 'same', strides=(2,2)),
        tf.keras.layers.Conv2DTranspose(32,(5,5), activation = 'elu', padding= 'same', strides=(2,2)),
        tf.keras.layers.Conv2DTranspose(16,(5,5), activation='elu', padding='same'),
        tf.keras.layers.Conv2DTranspose(1,(5,5), activation='elu', padding='same', strides=(2,2))
    ]
)




encoder.compile(optimizer='adadelta',
                loss=tf.losses.MeanAbsoluteError(),
                metrics=['acc'])

encoder.summary()


history = encoder.fit(x_train,
                            epochs=150,
                            batch_size=32,
                            shuffle=True,
                            validation_data=(x_test,x_test))
                


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



decoded_imgs = encoder.predict(x_val)


n = 2
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(40, 200))

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(40, 200))

plt.show()
