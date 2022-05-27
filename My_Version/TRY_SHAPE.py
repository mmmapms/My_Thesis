from cv2 import dnn_Model
from epftoolbox.models import DNN
from epftoolbox.models import DNNModel
from epftoolbox.data import DataScaler
import numpy as np
import pandas as pd
import time
import pickle as pc
import os
import shap

import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.data import scaling
from epftoolbox.data import read_data
import os
from sklearn.model_selection import train_test_split



# Set up the paths for saving data (this are the defaults for the library)
df = pd.read_csv('C:/Users/mmmap/epftoolbox/datasets/2015_NP.csv')


Y=df[' Price [EUR/MWh]']
X=df[df.columns.difference([' Price [EUR/MWh]', 'Date'])]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)


nVal = int(0.25 * X_train.shape[0])
nTrain = X_train.shape[0] - nVal # complements nVal

Xval = X_train[nTrain:] # last nVal obs
Xtrain = X_train[:nTrain] # first nTrain obs
Yval = Y_train[nTrain:]
Ytrain = Y_train[:nTrain]

"""
Xtrain = X_train.values
Xtest = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values

scaler = DataScaler('Norm')

Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.fit_transform(Xtest)
#Y_train_scaled = scaler.fit_transform(Y_train)
#Y_test_scaled = scaler.fit_transform(Y_test)
"""
"""
Changed the path of the path_hyperparameter_folder because it was not finding it
"""

neurons=[2]
n_features=8

model = DNNModel(neurons=neurons, n_features=n_features )
print(type(model))

model.fit(Xtrain.to_numpy(), Ytrain.to_numpy(), Xval.to_numpy(), Yval.to_numpy())

ex = shap.DeepExplainer(model.predict, X_train)

shap_values = ex.shap_values(X_test.iloc[0,:])


print(shap_values)

shap.summary_plot(shap_values, X_test)