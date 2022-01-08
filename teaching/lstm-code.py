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
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


"""
convert an array of values into a matrix
"""
def gen_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


"""
prepare the dataset
"""
def prepare_dataset(look_back=1):
    ### fix the random seed for reproducibility
    numpy.random.seed(42)

    ### load the dataset
    dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    ### normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    ### split the data into training set and test set
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    ### reshape into X=t and Y=t+1
    x_train, y_train = gen_dataset(train, look_back)
    x_test, y_test = gen_dataset(test, look_back)

    ### reshape the input to be [samples, time steps, features]
    x_train = numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = numpy.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    return x_train, y_train, x_test, y_test, dataset, scaler


"""
main program
"""
def run_lstm():
    ### prepare the dataset
    x_train, y_train, x_test, y_test, dataset, scaler = prepare_dataset()

    ### build, complie, and train the LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2)

    ### make predictions
    p_train = model.predict(x_train)
    p_test = model.predict(x_test)

    ### invert the predictions (in [0,1]) to the original range
    p_train = scaler.inverse_transform(p_train)
    y_train = scaler.inverse_transform([y_train])
    p_test = scaler.inverse_transform(p_test)
    y_test = scaler.inverse_transform([y_test])

    ### compute the RMSE
    trainScore = math.sqrt(mean_squared_error(y_train[0], p_train[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[0], p_test[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


if __name__ == "__main__":
    run_lstm()









