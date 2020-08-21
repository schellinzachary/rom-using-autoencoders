"""
Autoencoder für einen 1-D Druckpuls
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt
import numpy as np

import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

d=np.zeros([25,500,500])
nu = np.arange(25)/2
for i in range(25):
	x = np.linspace(0.0, 10.0, 500)
	d[i,:,:] = np.exp(x ** 2 - i/2)

	
d_train = (d - np.amin(d)) / (np.amax(d) - np.amin(d))
d_train = d_train.reshape((len(d_train),np.prod(d_train.shape[1:])))
train_ds = tf.constant(d_train)
target_ds = tf.constant(nu)

FEATURES = 250000
N_VALIDATION = 5
N_TRAIN = 20
BUFFER_SIZE = 25
BATCH_SIZE = 5
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE



lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	0.001,
	decay_steps=STEPS_PER_EPOCH*1000,
	decay_rate=1,
	staircase=False)

def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
	return[
		tfdocs.modeling.EpochDots(),
		tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',patience=200),
		tf.keras.callbacks.TensorBoard(logdir/name),
	]
def compile_and_fit(model, name, optimizer=None, max_epochs=500):
	if optimizer is None:
		optimizer = get_optimizer()
	model.compile(optimizer=optimizer,
				  loss=tf.keras.losses.MeanSquaredError(),
				  metrics=[
				  	tf.keras.losses.MeanSquaredError(name='mean_squared_error'),
				  	'accuracy'])
	model.summary()

	history = model.fit(
		train_ds,target_ds,
		steps_per_epoch=STEPS_PER_EPOCH,
		epochs=max_epochs,
		batch_size=BATCH_SIZE,
		validation_split=0.2,
		shuffle=False,
		callbacks=get_callbacks(name))
	return history

tiny_model = tf.keras.Sequential([
	layers.Dense(100, activation='elu', input_shape=(FEATURES,)),
	layers.Dense(100, activation='elu'),
	layers.Dense(1, activation='elu')
])


tiny_model.predict(train_ds)

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/tiny')
plotter = tfdocs.plots.HistoryPlotter(metric = 'mean_squared_error', smoothing_std=10)
plotter.plot(size_histories)
plt.show()


plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.xlabel("Epochs [Log Scale]")
plt.show()