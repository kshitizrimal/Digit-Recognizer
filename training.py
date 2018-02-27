import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# List of files
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Import Keras and its methods
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

# Normalize input
mean = np.std(X_train)
X_train -= mean
X_test -= mean

# Get input shape and number of classes
input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Simple 2-Dense Layer Keras model with 2 different dropout rate
# Dropout rates high at first will have negative impact on model as it will help to lose information
# so make it less at first and bigger at later layers
model = Sequential()
model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# categorical loss and Adam as the optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['acc'])

# Training 20 epochs with 0.1 val-train split and batch-size as 25
print("Training...")
model.fit(X_train, y_train, epochs=20, batch_size=25, validation_split=0.1)

# Save prediction on variable
print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

# Function to save result to a file
def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

# Write to file your test score for submission
write_preds(preds, "kaggle_submit.csv")