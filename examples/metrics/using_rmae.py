from epftoolbox.evaluation import rMAE
from epftoolbox.data import read_data
import pandas as pd

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecast = pd.read_csv('C:/Users/mmmap/Desktop/Thesis/Results/Values_Forecast_BE_2015_2_fea_DNN_LEAR.csv', index_col=0)

# Transforming indices to datetime format
forecast.index = pd.to_datetime(forecast.index)

# Reading data from the NP market
_, df_test = read_data(path='C:/Users/mmmap/epftoolbox/datasets/', dataset='BE_2015_2_fea', begin_test_date='04/01/2020', 
                       end_test_date='31/12/2021')

# Extracting forecast of DNN ensemble and display
fc_DNN_ensemble = forecast.loc[:, ['DNN']]

# Extracting real price and display
real_price = df_test.loc[:, ['Price']]

# Building the same datasets with shape (ndays, n_prices/day) instead 
# of shape (nprices, 1) and display
fc_DNN_ensemble_2D = pd.DataFrame(fc_DNN_ensemble.values.reshape(-1, 24), 
                                  index=fc_DNN_ensemble.index[::24], 
                                  columns=['h' + str(hour) for hour in range(24)])
real_price_2D = pd.DataFrame(real_price.values.reshape(-1, 24), 
                             index=real_price.index[::24], 
                             columns=['h' + str(hour) for hour in range(24)])
fc_DNN_ensemble_2D.head()


# According to the paper, the rMAE of the DNN ensemble for the NP market is 0.403
# when m='W'. Let's test the metric for different conditions

# Evaluating rMAE when real price and forecasts are both dataframes
rMAE(p_pred=fc_DNN_ensemble, p_real=real_price)

# Evaluating rMAE when real price and forecasts are both numpy arrays
a=rMAE(p_pred=fc_DNN_ensemble.values, p_real=real_price.values, m='W', freq='1H')

# Evaluating rMAE when input values are of shape (ndays, n_prices/day) instead 
# of shape (nprices, 1)
# Dataframes
b=rMAE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D, m='W')
# Numpy arrays
c=rMAE(p_pred=fc_DNN_ensemble_2D.values, p_real=real_price_2D.values, m='W', freq='1H')

# Evaluating rMAE when input values are of shape (nprices,) 
# instead of shape (nprices, 1)
# Pandas Series
d=rMAE(p_pred=fc_DNN_ensemble.loc[:, 'DNN'], 
     p_real=real_price.loc[:, 'Price'], m='W')
# Numpy arrays
e=rMAE(p_pred=fc_DNN_ensemble.values.squeeze(), 
     p_real=real_price.values.squeeze(), m='W', freq='1H')

print(a,b,c,d,e)
# We can also test situations where the rMAE will display errors

# Evaluating rMAE when real price and forecasts are of different type (numpy.ndarray and pandas.DataFrame)
rMAE(p_pred=fc_DNN_ensemble.values, p_real=real_price, m='W', freq='1H')

# Evaluating rMAE when real price and forecasts are of different type (pandas.Series and pandas.DataFrame)
rMAE(p_pred=fc_DNN_ensemble, p_real=real_price.loc[:, 'Price'])

# Evaluating rMAE when real price and forecasts are both numpy arrays of different size
rMAE(p_pred=fc_DNN_ensemble.values[1:], p_real=real_price.values, m='W')

# Evaluating rMAE when real price and forecasts are both dataframes are of different size
rMAE(p_pred=fc_DNN_ensemble.iloc[1:, :], p_real=real_price)

# Evaluating rMAE when real price are not multiple of 1 day
rMAE(p_pred=fc_DNN_ensemble.values[1:], p_real=real_price.values[1:], m='W', freq='1H')