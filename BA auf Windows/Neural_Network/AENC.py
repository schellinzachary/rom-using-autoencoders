'''
Concolutional Autoencoder 1.3
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io as sio
#load data
f1 = sio.loadmat('/home/zachary/BA/data_sod/sod241Kn0p00001/f.mat')
f2 = sio.loadmat('/home/zachary/BA/data_sod/sod25Kn0p00001/f.mat')
f3 = sio.loadmat('/home/zachary/BA/data_sod/sod25Kn0p01/f.mat')

f1 = f1['f']
f2 = f2['f']
f3 = f3['f']

#Getting dimensions 
shape = f1.shape
t = shape[0] 
v = shape[1] 
x = shape[2] 
#data prepocessing
BATCH_SIZE = 10
TRAIN_BUF = 100
train_dataset = f1
val_dataset = f2
train_dataset = (train_dataset - np.amin(train_dataset))/(np.amax(train_dataset)-np.amin(train_dataset))
val_dataset = (val_dataset- np.amin(val_dataset))/(np.amax(val_dataset)-np.amin(val_dataset))
train_dataset = tf.reshape(train_dataset,[t,v*x,1])
val_dataset = tf.reshape(val_dataset,[25,v*x,1])

#train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(TRAIN_BUF).batch(BATCH_SIZE)


encoding_dim=(x*v)/2

#Architecture definition
autoencoder = tf.keras.Sequential()
			  #tf.keras.Input(shape=(v*x,1)),
autoencoder.add(layers.Dense(encoding_dim, input_shape=(v*x,1),activation='relu'))
autoencoder.add(layers.Dense(v*x, activation='relu'))



autoencoder.compile(
	optimizer= tf.keras.optimizers.Adadelta(),
	loss=tf.keras.losses.MeanAbsolutePercentageError(),
	metrics=['acc'])

history = autoencoder.fit(
			train_dataset,train_dataset,
			epochs=1,
			batch_size = 4,
			shuffle=True,
			validation_data=(val_dataset,val_dataset))

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
	
