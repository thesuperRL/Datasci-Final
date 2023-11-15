import numpy as np 
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
# help with this https://github.com/eli5-org/eli5/issues/39
import eli5

#overall a lot of places here where I used Keras and ELI5 Documentation

# two major tutorials that helped me 
# https://www.kdnuggets.com/2018/06/basic-keras-neural-network-sequential-model.html
# https://keras.io/guides/sequential_model/

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

def get_basic_model():
  model = tf.keras.Sequential([
    Dense(4, activation='relu'),
    Dense(8, activation='relu'),
    Dropout(0.125),
    Dense(1, activation='relu'),
  ])

  # used MSLE because of this https://stackoverflow.com/questions/46014723/keras-extremely-high-loss

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['mean_squared_logarithmic_error', 'mean_absolute_error'])
  return model

model = get_basic_model()
model.fit(x_train, y_train, epochs=18, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score[0])
print('Test MSLE:', score[1])

# code and help from here https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python
perm = eli5.sklearn.PermutationImportance(model, random_state=1, scoring="neg_mean_squared_error").fit(x_test,y_test)
with open("result.html", "w") as file:
    file.write(eli5.show_weights(perm, feature_names = preprocessed.columns.tolist()).data)