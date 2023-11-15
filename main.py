import numpy as np 
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import eli5


BATCH_SIZE = 15

preprocessed = pd.read_csv("cleaned_data")
preprocessed.drop(columns=preprocessed.columns[0], axis=1, inplace=True)

x_train, x_test = train_test_split(preprocessed, test_size=0.2)

y_train = x_train.pop('crime_rate')
y_test = x_test.pop('crime_rate')

x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

print(preprocessed)
print(x_train)

preprocessed.pop('crime_rate')

tf.convert_to_tensor(x_train)
tf.convert_to_tensor(x_test)

# normalizer = Normalization(axis=-1)
# normalizer.adapt(preprocessed)

def get_basic_model():
  model = tf.keras.Sequential([
    #normalizer,
    Dense(4, activation='relu'),
    Dense(8, activation='relu'),
    Dropout(0.125),
    Dense(1, activation='relu'),
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['mean_squared_logarithmic_error', 'mean_absolute_error'])
  return model

# model = Sequential()

# model.add(Dense(32, activation='relu', input_shape=(10, )))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='relu'))

model = get_basic_model()
model.fit(x_train, y_train, epochs=18, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score[0])
print('Test MSLE:', score[1])

perm = eli5.sklearn.PermutationImportance(model, random_state=1, scoring="neg_mean_squared_error").fit(x_test,y_test)
with open("result.html", "w") as file:
    file.write(eli5.show_weights(perm, feature_names = preprocessed.columns.tolist()).data)