from epftoolbox.evaluation import MAE, sMAPE, rMAE, MAPE
from epftoolbox.data import read_data
import pandas as pd

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecast = pd.read_csv('C:/Users/mmmap/Desktop/Thesis/Results/LEAR_FINAL/Results_2015_NP_no_wind_CW365_YT6.csv', index_col=0)



# Transforming indices to datetime format
forecast.index = pd.to_datetime(forecast.index)


# Reading data from the NP market
_, df_test = read_data(path='C:/Users/mmmap/epftoolbox/datasets/', dataset='2015_NP_no_wind', begin_test_date='09/01/2016', 
                       end_test_date='31/12/2021')


print(forecast)
# Extracting forecast of DNN ensemble and display
fc_DNN_ensemble = forecast.loc[:, ['LEAR']]

# Extracting real price and display
real_price = df_test.loc[:, ['Price']]

# Building the same datasets with shape (ndays, n_prices/day) 
# instead of shape (nprices, 1) and display
fc_DNN_ensemble_2D = pd.DataFrame(fc_DNN_ensemble.values.reshape(-1, 24), 
                                  index=fc_DNN_ensemble.index[::24], 
                                  columns=['h' + str(hour) for hour in range(24)])
real_price_2D = pd.DataFrame(real_price.values.reshape(-1, 24), 
                             index=real_price.index[::24], 
                             columns=['h' + str(hour) for hour in range(24)])
fc_DNN_ensemble_2D.head()


# According to the paper, the MAE of the DNN ensemble for the NP market is 1.667

#print(fc_DNN_ensemble, real_price)
# Evaluating MAE when real price and forecasts are both numpy arrays
a=MAE(p_pred=fc_DNN_ensemble.values, p_real=real_price.values)

b=rMAE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D, m='W')

c=sMAPE(p_pred=fc_DNN_ensemble.values, p_real=real_price.values) * 100

d=MAPE(p_pred=fc_DNN_ensemble_2D.values, p_real=real_price_2D.values) * 100

print('MAE: {:.3f}  |  rMAE: {:.3f}  |  sMAPE: {:.3f}   |  MAPE: {:.3f}'.format(a,b,c,d))
# We can also test situations where the MAE will display errors

