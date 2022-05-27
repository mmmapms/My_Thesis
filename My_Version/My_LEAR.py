"""
Simplified example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

from epftoolbox.models import LEAR
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
from statsmodels.robust import mad
import os

from sklearn.linear_model import LassoLarsIC, Lasso
from epftoolbox.data import scaling
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
dataset = 'Full_Dataset-NP'

# Number of years (a year is 364 days) in the test dataset.
years_test = 4

# Number of days used in the training dataset for recalibration
calibration_window = 90

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

path_datasets_folder = os.path.join('.', 'C:/Users/mmmap/epftoolbox/datasets')
path_recalibration_folder = os.path.join('.', 'experimental_files')

if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)

# Defining train and testing data
df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                begin_test_date=begin_test_date, end_test_date=end_test_date)



# Defining unique name to save the forecast
forecast_file_name = 'LEAR_forecast' + '_dat' + str(dataset) + '_YT' + str(years_test) + \
                        '_CW' + str(calibration_window) + '.csv'

forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)


# Defining empty forecast array and the real values to be predicted in a more friendly format
forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

forecast_dates = forecast.index

model = LEAR(calibration_window=calibration_window)
results={"smape":[],
            "mae":[]}
# For loop over the recalibration dates
for date in forecast_dates:

    # For simulation purposes, we assume that the available data is
    # the data up to current date where the prices of current date are not known
    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)


    #data_available = data_available.reset_index()

    # We set the real prices for current date to NaN in the dataframe of available data
    data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN


    # Recalibrating the model with the most up-to-date available data and making a prediction
    # for the next day
    Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)
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

print('Total mean:\nsMAPE: {:.2f}%  |  MAE: {:.3f}'.format(np.mean(results["smape"]), np.mean(results["mae"])))