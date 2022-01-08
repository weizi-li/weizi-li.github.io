### enforce to use CPU only
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### import auxiliary libraries
import numpy
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

### import keras and relevant functions
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


"""
save a trained model for future use
"""
def save_model(model):
    ### serialize model to JSON (YAML): save the model architecture
    model_json = model.to_json()
    with open("model-architecture.json", "w") as json_file:
        json_file.write(model_json)

    ### serialize weights to HDF5: save the model weights
    model.save_weights("model-weights.h5")
    print("Model saved!")


"""
design the architecture of a model
"""
def design_model(input_shape, num_classes):
    model = Sequential()
    ### https://keras.io/api/layers/convolution_layers/convolution2d/
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),  # 32 3x3 filters -> 32 feature/activation maps
                     strides=(1, 1),
                     padding="valid",
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


"""
model training 
"""
def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(loss=keras.losses.categorical_crossentropy,  # https://keras.io/api/losses/
                  optimizer=keras.optimizers.SGD(),  # https://keras.io/api/optimizers/
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])
    return model


"""
use an existing model 
"""
def use_existing_model(x_test, y_test):
    ### load a model
    json_file = open('model-architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    ### load weights
    loaded_model.load_weights("model-weights.h5")

    ### compile the model
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])

    ### apply the model
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])


"""
main program 
"""
def cnn():
    ### load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    ### initialization
    batch_size = 128
    num_classes = 10
    epochs = 1

    ### input image dimensions
    img_rows, img_cols = 28, 28

    ### specify the input_shape
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    print(input_shape)

    ### normalize the input
    x_train = x_train.astype('float32') # Copy of the array, cast to a specified type.
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    ### convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ### design, train, and save a CNN model
    # initial_model = design_model(input_shape, num_classes)
    # trained_model = train_model(initial_model, x_train, y_train, x_test, y_test, batch_size, epochs)
    # save_model(trained_model)

    ### use a previously saved CNN model
    use_existing_model(x_test, y_test)


if __name__ == "__main__":
    cnn()










