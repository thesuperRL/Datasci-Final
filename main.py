import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

x = tf.ones((50, 10))

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(10, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='relu'))

y = model(x)

print(model.weights)

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

print(model.summary())