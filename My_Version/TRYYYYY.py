"""
Simplified example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

from cv2 import mean
import pip
from tensorboard import summary
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

feat_importance=True

feat_keywords=['Price',' Generation forecast (France)',' System load forecast (FR)']
exog_features=['Price',' Generation forecast (France)',' System load forecast (FR)']

#feat_keywords=["Price"," Generation forecast (FR)"," System load forecast (FR) (MW)","Day-Ahead Generation forecast wind (BE) (MWh)","Day-ahead generation forecast Solar (BE) (MWh)","Grid load forecast (BE) (MW)"]
#exog_features=["Price"," Generation forecast (FR)"," System load forecast (FR) (MW)","Day-Ahead Generation forecast wind (BE) (MWh)","Day-ahead generation forecast Solar (BE) (MWh)","Grid load forecast (BE) (MW)"]

#feat_keywords=["Price"," Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (EE) [MWh]","Wind power forecast (LT) [MWh]","Wind power forecast (SE) [MWh]","Wind power forecast (LV) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]"]
#exog_features=["Price"," Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (EE) [MWh]","Wind power forecast (LT) [MWh]","Wind power forecast (SE) [MWh]","Wind power forecast (LV) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]"]
lags=[1,2]

results={"date":[], "smape":[], "mae":[], "hourly":[]}

dataset = 'BE_2015_2_fea'

# Number of years (a year is 364 days) in the test dataset.
years_test = 0.0002

# Number of days used in the training dataset for recalibration
calibration_window = 20

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = '20/12/2021'
end_test_date = '25/12/2021'

path_datasets_folder = os.path.join('.', 'C:/Users/mmmap/epftoolbox/datasets')
path_recalibration_folder = os.path.join('.', 'C:/Users/mmmap/epftoolbox/Final_Out')

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

feat_imp_dict={}

if feat_importance:
    if isinstance(feat_keywords,str):
        if feat_keywords in ["","total","all"]:
            pass
        elif len(feat_keywords)==1:
            if feat_keywords[0] in ["","total","all"]:
                pass
        else:
            assert(feat_keywords is not None)
            assert(exog_features is not None)
    for keyword in feat_keywords:
        if keyword in ["","total","all"]:
            keyword="total"
        feat_imp_dict[str(keyword)+"_coef_sum"]=[]

    results["const_features"]=[]
    results["coef_sum"]=[]
    results["coef"]=[]

if lags is None:
    lags=[]
# Calculate number of lagged hourly features generated from the exgoneous features
if exog_features is not None:
    if isinstance(lags[0],int):
        feat_multip=(len(lags)+1)*24
    else:
        assert(isinstance(lags[0],list))
        feat_multip=[]
        for feature_lags in lags:
            feat_multip.append((len(feature_lags)+1)*24)
else:
    feat_multip=None

model = LEAR(calibration_window=calibration_window, lags=lags)

def impute_coef(coef_list,deleted_list):
    
    coef_fixed=coef_list
    for deleted in deleted_list:
        coef_fixed.insert(deleted, 0)
    return coef_fixed
    
def sum_keyword_coefs(features,keyword,coef_list_hour,price_mult=96,feat_mult=72):
  '''
  Method for summing up hourly coefficients for different feature groups
  (eg. all hourly features built from features containing "solar" in their
  name). In order for this to work the hourly features need to be in the 
  following order:
  1. 24 hourly features each from Price d, d-t1, d-t2, etc
  2. 24 hourly features each from Exogenous 1 d, d-t1, d-t2, etc
  3. 24 hourly features each from Exogenous 2 d, d-t1, d-t2, etc
  4. so on and so forth

  Parameters
  ----------
  features : list of strings
      List of original feature names before they got renamed to Exogenous 1, 2,
      ..., N
  keyword : str
      Keyword to look for in the original feature names. If all coefficients are
      to be summed up any of "", "total" or "all" will work.
  coef_list: list of floats
      List of all coefficients (with constant features being imputed back as
      having coefficient 0).
  price_mult: int, optional
      Number of hourly features built from Price. By default this is 4*24=96.
  feat_mult: int or list of ints, optional
      Number of hourly features built from Exogenous features. By default
      this is 3*24 for all exog. features. As the number of built lagged
      features can differ for the exog. features, this can also be a list with
      the multipliers for the different exog. features.
  
  '''
  features_aux=features[1:]
  # if feature multiplier is the same for all features
  
  if isinstance(feat_mult,int):
    sum=0
    if keyword=="Price":
      for l in range(0,96):
        sum+=coef_list_hour[l]
    else:
      # go over features
      for i in range(len(features_aux)):
        # if feature has keyword
        if keyword in features_aux[i]:
          # sum all lagged hourly features corresponding to it
          for j in range(feat_mult):
            index=price_mult+i*feat_mult+j
            sum+=coef_list_hour[index]

  #n_features = 96 + (len(data_available.columns) - 1) * 72 + 7

  return sum

def predict(X):

    # Predefining predicted prices
    Yp = np.zeros(24)
    b_val = np.zeros(24)
    Coef = []


    # # Rescaling all inputs except dummies (7 last features)
    X_no_dummies = model.scaler_X.transform(X[:, :-7])
    X[:, :-7] = X_no_dummies

    # Predicting the current date using a recalibrated LEAR
    for h in range(24):
        # Predicting test dataset and saving
        Yp[h] = model.models[h].predict(X)
        Coef.append(model.models[h].coef_)
        b_val[h] = model.models[h].intercept_


    print(np.sum(np.array(X)*np.array(Coef[0])))

    Yp = model.scaler_Y.inverse_transform(Yp.reshape(1, -1))
    b_val = model.scaler_Y.inverse_transform(b_val.reshape(1, -1))
    return Yp

def recalibrate_predict(X_train, Y_train, X_test):

    model.recalibrate(X_train=X_train, Y_train=Y_train)

    Yp = predict(X=X_test)

    return Yp

def recalibrate_and_forecast_next_day(df, next_day_date):

    # We define the new training dataset and test datasets 
    df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
    # Limiting the training dataset to the calibration window
    df_train = df_train.iloc[-calibration_window * 24:]

    # We define the test dataset as the next day (they day of interest) plus the last two weeks
    # in order to be able to build the necessary input features. 
    df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

    # Generating X,Y pairs for predicting prices
    X_train, Y_train, X_test, = model._build_and_split_XYs(
        df_train=df_train, df_test=df_test, date_test=next_day_date)

    # Recalibrating the LEAR model and extracting the prediction
    Yp = recalibrate_predict(X_train=X_train, Y_train=Y_train, X_test=X_test)

    return Yp

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
    Yp = recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)
  
    forecast.loc[date, :] = Yp

    # Saving forecast
    forecast.to_csv(forecast_file_path)

    results["const_features"].append(model.const_features)

    tot_coef=[]

    X_test, Y_train=model.x_maninha(df=data_available,next_day_date=date)
    df=pd.DataFrame(X_test)
    df.to_csv('C:/Users/mmmap/epftoolbox/Final_Out/xtest_no.csv')

    Pred=np.average(np.array(Yp))
    Real=np.average(np.array(sum(Y_train,0))/24)

    print('Mean Prediction: {:.3f}  |  Mean Real Price: {:.3f}'.format(np.array(Pred), np.array(Real)))

    looo=[]

    for i in range(24):

      coef_fixed_hourly=np.array(impute_coef(list(model.models[i].coef_),results["const_features"][-1]))
      Xtest_fixed=np.array(impute_coef(list(X_test[0]),results["const_features"][-1]))
      looo.append(np.sum(coef_fixed_hourly*Xtest_fixed))
      print(np.sum(coef_fixed_hourly*Xtest_fixed))
      for keyword in feat_keywords:
        # summing up coefs for a given keyword
        tot_coef.append(sum_keyword_coefs(exog_features,str(keyword),coef_fixed_hourly,96,feat_multip))
        # if no recalibration just copy the previous
      results["coef"].append(tot_coef)
      tot_coef=[]

df=pd.DataFrame(results["coef"])

df.columns=feat_keywords
df.insert(0, 'Dates', df_test.index)

df.to_csv('C:/Users/mmmap/epftoolbox/Final_Out/Coef_2015_NP.csv', index = False)
print('Total mean:\nsMAPE: {:.2f}%  |  MAE: {:.3f}'.format(np.mean(results["smape"]), np.mean(results["mae"])))

