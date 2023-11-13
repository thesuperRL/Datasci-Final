import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

class Model:
    def __init__(self, train):
        model = Sequential()

        model.add(Dense(128, activation='softmax', input_dim = train.shape[1]))
        model.add(Dense(256, activation='softmax'))
        model.add(Dense(256, activation='softmax'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='softmax'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='softmax'))
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

        print(model.summary())