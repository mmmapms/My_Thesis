"""
Simplified example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

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
feat_keywords=[" Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (SE) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]", "all"]

exog_features=[" Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (SE) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]"]
lags=[1,2]





results={"date":[], "smape":[], "mae":[], "hourly":[]}

dataset = '2015_NP_no_wind'



# Number of years (a year is 364 days) in the test dataset.
years_test = 7

# Number of days used in the training dataset for recalibration
calibration_window = 90

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = '02/04/2015'
end_test_date = '31/12/2021'

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
    
def add_coef (features,keyword,coef_list,price_mult=96,feat_mult=72):
  # shortcut in case all coefficients need to be summed up

  a = [0 for i in range(23)] 
  for i in range(23):

    if keyword in ["","total","all"]:
        a[i]=0
        for coef in coef_list[i]:
            a[i]+=coef
        return a
  # if feature multiplier is the same for all features
    if isinstance(feat_mult,int):
        a[i]=0
    # go over features
        for i in range(len(features)):
      # if feature has keyword
            if keyword in features[i]:
        # sum all lagged hourly features corresponding to it
                for j in range(feat_mult):
    
                    index=price_mult+i*feat_mult+j

                    a[i]+=np.abs(coef_list[i][index])
  # if feature multiplier is different for the features
    else:
        assert(isinstance(feat_mult,list))
        assert(len(feat_mult)==len(features))
        a[i]=0
    # go over features
        for i in range(len(features)):
      # if feature has keyword
            if keyword in features[i]:
        # sum all lagged hourly features corresponding to it
                for j in range(feat_mult[i]):
                    index=price_mult+i*feat_mult[i]+j
                    a[i]+=np.abs(coef_list[index][i])
  #n_features = 96 + (len(data_available.columns) - 1) * 72 + 7
    
    return a    

def sum_keyword_coefs(features,keyword,coef_list,price_mult=96,feat_mult=72):
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
  # shortcut in case all coefficients need to be summed up
  if keyword in ["","total","all"]:
    sum=0
    for coef in coef_list:
      sum+=coef
    return sum
  # if feature multiplier is the same for all features
  if isinstance(feat_mult,int):
    sum=0
    # go over features
    for i in range(len(features)):
      # if feature has keyword
      if keyword in features[i]:
        # sum all lagged hourly features corresponding to it
        for j in range(feat_mult):
          index=price_mult+i*feat_mult+j
          sum+=np.abs(coef_list[index])
  # if feature multiplier is different for the features
  else:
    assert(isinstance(feat_mult,list))
    assert(len(feat_mult)==len(features))
    sum=0
    # go over features
    for i in range(len(features)):
      # if feature has keyword
      if keyword in features[i]:
        # sum all lagged hourly features corresponding to it
        for j in range(feat_mult[i]):
          index=price_mult+i*feat_mult[i]+j
          sum+=np.abs(coef_list[index])
  #n_features = 96 + (len(data_available.columns) - 1) * 72 + 7

  return sum

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

    
    if feat_importance:
      # at every calibration some features get deleted
      if True:
        results["const_features"].append(model.const_features)
        # sum up the absolute value of coefficients for every feature
        #results["coef_sum"].append(np.abs(model.models[0].coef_))
        results["coef_sum"].append(model.models[0].coef_)
        
        coef_fixed_hourly=impute_coef(list(model.models[0].coef_),results["const_features"][-1])
        results["hourly"].append(coef_fixed_hourly)
        for i in range(23):
          #results["coef_sum"][-1]=results["coef_sum"][-1]+np.abs(model.models[i+1].coef_)
          results["coef_sum"][-1]=results["coef_sum"][-1]+model.models[i+1].coef_
          coef_fixed_hourly=impute_coef(list(model.models[i+1].coef_),results["const_features"][-1])

          results["hourly"].append(coef_fixed_hourly)
        # impute deleted features as having coef 0
        coef_fixed=impute_coef(list(results["coef_sum"][-1]),results["const_features"][-1])

        
        # Calculate feature group importance for original features
        for keyword in feat_keywords:
          if keyword in ["","total","all"]:
            keyword="total"
          # summing up coefs for a given keyword
          feat_imp_dict[str(keyword)+"_coef_sum"].append(sum_keyword_coefs(exog_features,str(keyword),coef_fixed,96,feat_multip))
          tot_coef=add_coef(exog_features,str(keyword),results["hourly"],96,feat_multip)
      # if no recalibration just copy the previous
      else:
        results["const_features"].append(results["const_features"][-1])
        results["coef_sum"].append(results["coef_sum"][-1])
        for keyword in feat_keywords:
          if keyword in ["","total","all"]:
            keyword="total"
          feat_imp_dict[str(keyword)+"_coef_sum"].append(feat_imp_dict[str(keyword)+"_coef_sum"][-1]) 
      

df=pd.DataFrame(results["hourly"])
coef_hour=pd.DataFrame()
index_aux=0
for keyword in exog_features:
    coef_hour[str(keyword)]=df.iloc[:,(index_aux*72+96):((96+(72*(index_aux+1)))-1)].sum(axis=1)
    index_aux+=1

  #n_features = 96 + (len(data_available.columns) - 1) * 72 + 7

coef_hour.insert(0, 'Dates', df_test.index)

#print(pd.DataFrame(results["coef_sum"][-1]))


coef_hour.to_csv('C:/Users/mmmap/epftoolbox/My_Version/LEAR_2015_NP_no_wind_Y7_Coef.csv', index = False)
print('Total mean:\nsMAPE: {:.2f}%  |  MAE: {:.3f}'.format(np.mean(results["smape"]), np.mean(results["mae"])))

