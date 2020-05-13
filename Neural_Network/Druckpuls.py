"""
Autoencoder für einen 1-D Druckpuls
"""

import tensorflow as tf
import numpy as np
d=tf.zeros(500,500, tf.int32)
for i in range(25):
	x = tf.linspace(0.0, 10.0, 500.0)
	d[:,:,i] = tf.math.exp(x ** 2 - i/2)

inp_shape = tf.shape(d)
d_train = (d - np.amin(d)) / (np.amax(d) - np.amin(d))
encoding_dim = 1

N_VALIDATION = 5
N_TRAIN = 20
BUFFER_SIZE = 25
BATCH_SIZE = 5
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE


lr_schedule = tf.keras.optimizers.shedules.InverseTimeDecay(
	0.001,
	decay_steps=STEPS_PER_EPOCH*1000,
	decay_rate=1,
	staircase=False)

def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)


