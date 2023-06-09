import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# Reading data
df = pd.read_csv('twitter_racism_parsed_dataset.csv')
pd.set_option("display.max.columns", None)
df.drop(['index', 'id', 'Annotation'], inplace=True, axis=1)

# Data Preprocessing
# Removing special characters
def removeSpecialCharacter(v):
    c = "".join([r for r in v if ('A' <= r <= 'Z') or ('a' <= r <= 'z') or (r == " ")])
    return c


df['Text']= df['Text'].apply(lambda x: removeSpecialCharacter (str(x)))

# Dropping any empty Text
df['Text'].replace('', np.nan, inplace=True)
df.dropna(subset=['Text'], inplace=True)

# Dropping any duplicated comment
df = df.drop_duplicates(subset='Text', keep='first')



X = df['Text']
y = df['oh_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=None)


trainList, testList = [], []

for i in X_train:
    trainList.append(i)

for i in X_test:
    testList.append(i)
    
#splitting data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Padding the data samples to the maximum post length
vect_train = pad_sequences(tokenizer.texts_to_sequences(trainList))
vect_test = pad_sequences(tokenizer.texts_to_sequences(testList))

# Build model on TPU hardware
with tpu_strategy.scope():
    
    # Initialize a sequential CNN model
    model = Sequential()
    
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))

    model.add(Conv1D(filters=60, kernel_size=len(tokenizer.word_index) + 1,
                     padding='same', activation='relu'))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(filters=60, kernel_size=len(tokenizer.word_index) + 1,
                     padding='same', activation='relu'))
    model.add(MaxPooling1D())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'], steps_per_execution=64)
    
    model.summary()


# Fitting the data onto model
model.fit(vect_train, y_train, validation_data=(vect_test, y_test),
          epochs=10, batch_size=1024, verbose=2)

# Getting the accuracy score
scores = model.evaluate(vect_test, y_test, verbose=2)

# Displaying the accuracy of correct prediction over test data
print("Accuracy: %.2f%%" % (scores[1] * 100))
