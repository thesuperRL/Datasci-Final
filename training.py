import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, SimpleRNN, Bidirectional,LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

class Model:
    def __init__(self):
        model = Sequential()

        model.add(Dense(10, activation='softmax'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='softmax'))

        model.summary()