import keras
from keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

x_train = np.load("data/small_input.npy").astype(np.float32)
y_train = np.load("data/small_output.npy").astype(np.float32)

model = load_model("model_2d.hf5")
print (model.summary())
n_train = 200
n_test = 100
x_test = x_train[n_train:n_train+n_test, :, :]
y_test = y_train[n_train:n_train+n_test, :, :]
print ('shape of x_test', x_test.shape)
print ('shape of y_test', y_test.shape)
print (np.sum(x_test))
print (np.sum(y_test))
N = x_train.shape[1]
model.evaluate(x_test.reshape(n_test,N,N,1), y_test.reshape(n_test,N,N,1), verbose=1)

# for layer in model.layers:
#     weights = layer.get_weights()
#     print (weights)