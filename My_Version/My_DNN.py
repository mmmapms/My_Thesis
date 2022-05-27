from epftoolbox.models import DNN
import numpy as np
import pandas as pd
import time
import pickle as pc
import os

import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K

from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.data import scaling
from epftoolbox.data import read_data
import os

# Number of layers in DNN
nlayers = 2

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
dataset = "2015_BE"

# Number of years (a year is 364 days) in the test dataset.
years_test = 6

# Boolean that selects whether the validation and training datasets were shuffled when
# performing the hyperparameter optimization. Note that it does not select whether
# shuffling is used for recalibration as for recalibration the validation and the
# training datasets are always shuffled.
shuffle_train = 1

# Boolean that selects whether a data augmentation technique for DNNs is used
data_augmentation = 0

# Boolean that selects whether we start a new recalibration or we restart an existing one
new_recalibration = 1

# Number of years used in the training dataset for recalibration
calibration_window = 1

# Unique identifier to read the trials file of hyperparameter optimization
experiment_id = 1

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

# Set up the paths for saving data (this are the defaults for the library)
path_wd="/content"
path_datasets_folder = os.path.join(path_wd, 'C:/Users/mmmap/epftoolbox/datasets')
path_recalibration_folder = os.path.join(path_wd, 'C:/Users/mmmap/epftoolbox/examples/experimental_files')
path_hyperparameter_folder = os.path.join(path_wd, 'C:/Users/mmmap/epftoolbox/examples/experimental_files')

if not os.path.exists(path_recalibration_folder):
    os.makedirs(path_recalibration_folder)

# Defining train and testing data
df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                begin_test_date=begin_test_date, end_test_date=end_test_date)
# Defining unique name to save the forecast

forecast_file_name = 'DNN_forecast_nl' + str(nlayers) + '_dat' + str(dataset) + \
                        '_YT' + str(years_test) + '_SFH' + str(shuffle_train) + \
                        '_DA' * data_augmentation + '_CW' + str(calibration_window) + \
                        '_' + str(experiment_id) + '.csv'

forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

# Defining empty forecast array and the real values to be predicted in a more friendly format
forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

# If we are not starting a new recalibration but re-starting an old one, we import the
# existing files and print metrics 
if not new_recalibration:
    # Import existinf forecasting file
    forecast = pd.read_csv(forecast_file_path, index_col=0)
    forecast.index = pd.to_datetime(forecast.index)

    # Reading dates to still be forecasted by checking NaN values
    forecast_dates = forecast[forecast.isna().any(axis=1)].index

    # If all the dates to be forecasted have already been forecast, we print information
    # and exit the script
    if len(forecast_dates) == 0:

        mae = np.mean(MAE(forecast.values.squeeze(), real_values.values))
        smape = np.mean(sMAPE(forecast.values.squeeze(), real_values.values)) * 100
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format('Final metrics', smape, mae))
    
else:
    forecast_dates = forecast.index
"""
Changed the path of the path_hyperparameter_folder because it was not finding it
"""
model = DNN(experiment_id=experiment_id, path_hyperparameter_folder='C:/Users/mmmap/epftoolbox/examples/experimental_files',
            nlayers=nlayers, dataset=dataset, years_test=years_test, 
            shuffle_train=shuffle_train, data_augmentation=data_augmentation, 
            calibration_window=calibration_window)

results={"smape":[],
         "mae":[]}
# For loop over the recalibration dates
for date in forecast_dates:

    # For simulation purposes, we assume that the available data is
    # the data up to current date where the prices of current date are not known
    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

    # We set the real prices for current date to NaN in the dataframe of available data
    data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

    # Recalibrating the model with the most up-to-date available data and making a prediction
    # for the next day
    Yp,Xtrain, Xval, Xtest, Ytrain, Yval = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)
    te = model.scalerX
    tt=model.scaler
    # Saving the current prediction
    forecast.loc[date, :] = Yp

    # Computing metrics up-to-current-date
    mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
    smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100
    results["smape"].append(smape)
    results["mae"].append(mae)

    # Pringint information
    print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

    # Saving forecast
    forecast.to_csv(forecast_file_path) 
print("------------------------------------")
print('Total mean:\nsMAPE: {:.2f}%  |  MAE: {:.3f}'.format(np.mean(results["smape"]), np.mean(results["mae"])))
