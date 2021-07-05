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
for i in range(25):
	x = np.linspace(0.0, 10.0, 500)
	d[i,:,:] = np.exp(x ** 2 - i/2)
	
d_train = (d - np.amin(d)) / (np.amax(d) - np.amin(d))
d_train = d_train.reshape((len(d_train),np.prod(d_train.shape[1:])))
packed_ds = tf.constant(d_train)
#packed_ds=d_train
packed_ds = tf.data.Dataset.from_tensor_slices(packed_ds)

FEATURES = 250000
N_VALIDATION = 5
N_TRAIN = 20
BUFFER_SIZE = 25
BATCH_SIZE = 5
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION) #.cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN) #.cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

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
		tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',patience=200),
		tf.keras.callbacks.TensorBoard(logdir/name),
	]
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
	if optimizer is None:
		optimizer = get_optimizer()
	model.compile(optimizer=optimizer,
				  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				  metrics=[
				  	tf.keras.losses.BinaryCrossentropy(
				  		from_logits=True, name='binary_crossentropy'),
				  	'accuracy'])
	model.summary()

	history = model.fit(
		train_ds,
		steps_per_epoch=STEPS_PER_EPOCH,
		epochs=max_epochs,
		validation_data=validate_ds,
		callbacks=get_callbacks(name),
		verbose=0)
	return history

tiny_model = tf.keras.Sequential([
	layers.Dense(100, activation='elu', input_shape=(FEATURES,)),
	layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/tiny')
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])