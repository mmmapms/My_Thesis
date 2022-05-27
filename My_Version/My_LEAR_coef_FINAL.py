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

#BE_2015_2_fea
#feat_keywords=['Price',' Generation forecast (France)',' System load forecast (FR)']
#exog_features=['Price',' Generation forecast (France)',' System load forecast (FR)']

#2015_BE
#feat_keywords=["Price"," Generation forecast (FR)"," System load forecast (FR) (MW)","Day-Ahead Generation forecast wind (BE) (MWh)","Day-ahead generation forecast Solar (BE) (MWh)","Grid load forecast (BE) (MW)"]
#exog_features=["Price"," Generation forecast (FR)"," System load forecast (FR) (MW)","Day-Ahead Generation forecast wind (BE) (MWh)","Day-ahead generation forecast Solar (BE) (MWh)","Grid load forecast (BE) (MW)"]

#2015_NP
#feat_keywords=["Price"," Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (EE) [MWh]","Wind power forecast (LT) [MWh]","Wind power forecast (SE) [MWh]","Wind power forecast (LV) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]"]
#exog_features=["Price"," Grid load forecast [MW]"," Wind power forecast (DK) [MWh]","Wind power forecast (EE) [MWh]","Wind power forecast (LT) [MWh]","Wind power forecast (SE) [MWh]","Wind power forecast (LV) [MWh]","Total Wind [MWh]","Hydro reservoir [GWh]"]

#2015_NP_no_wind1
#feat_keywords=['Price',' Grid load forecast [MW]','Wind power forecast (SE) [MWh]','Total Wind [MWh]','Hydro reservoir [GWh]']
#exog_features=['Price',' Grid load forecast [MW]','Wind power forecast (SE) [MWh]','Total Wind [MWh]','Hydro reservoir [GWh]']

#Full_Dataset-NP
feat_keywords=['Price',' Grid load forecast',' Wind power forecast']
exog_features=['Price',' Grid load forecast',' Wind power forecast']


#2015_NP_no_wind
feat_keywords=['Price',' Grid load forecast [MW]',' Wind power forecast (DK) [MWh]','Wind power forecast (SE) [MWh]','Total Wind [MWh]','Hydro reservoir [GWh]']
exog_features=['Price',' Grid load forecast [MW]',' Wind power forecast (DK) [MWh]','Wind power forecast (SE) [MWh]','Total Wind [MWh]','Hydro reservoir [GWh]']




lags=[1,2]

results={"date":[], "smape":[], "mae":[], "hourly":[]}

dataset = '2015_NP_no_wind'

# Number of years (a year is 364 days) in the test dataset.
years_test = 6

# Number of days used in the training dataset for recalibration
calibration_window = 365

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = None
end_test_date = None

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
    results["coefXx_test"]=[]
    results["intercept"]=[]
    results["LEAR"]=[]

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

# For loop over the recalibration dates
for date in forecast_dates:
    print(date)
    # For simulation purposes, we assume that the available data is
    # the data up to current date where the prices of current date are not known
    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)


    #data_available = data_available.reset_index()

    # We set the real prices for current date to NaN in the dataframe of available data
    data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

    # Recalibrating the model with the most up-to-date available data and making a prediction
    # for the next day
    Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)
    
    results["LEAR"].append(Yp[0])

    forecast.loc[date, :] = Yp

    # Saving forecast
    forecast.to_csv(forecast_file_path)

    results["const_features"].append(model.const_features)

    sum_coef_feat=[]
    sum_coefXx_test_feat=[]
    intercept=[]

    X_test=model.x_maninha(df=data_available,next_day_date=date)

    for i in range(24):

      coef_fixed_hourly=np.array(impute_coef(list(model.models[i].coef_),results["const_features"][-1]))
      Xtest_fixed=np.array(impute_coef(list(X_test[0][0]),results["const_features"][-1]))
      CoefXx_test=coef_fixed_hourly*Xtest_fixed
      for keyword in feat_keywords:
        # summing up coefs for a given keyword
        sum_coef_feat.append(sum_keyword_coefs(exog_features,str(keyword),coef_fixed_hourly,96,feat_multip))
        sum_coefXx_test_feat.append(sum_keyword_coefs(exog_features,str(keyword),CoefXx_test,96,feat_multip))
    
        # if no recalibration just copy the previous
      results["coef"].append(sum_coef_feat)
      results['coef_sum'].append(np.sum(np.abs(np.array(sum_coef_feat))))
      results["coefXx_test"].append(sum_coefXx_test_feat)
      results["intercept"].append(model.models[i].intercept_)
      sum_coef_feat=[]
      sum_coefXx_test_feat=[]
      intercept=[]

sum_abs=np.sum(np.array(results['coef_sum']))

#BE_2015_2_fea
#Title_Coef=['Coef Price','Coef Generation forecast (FR)','Coef Load forecast (FR)']
#Title_Coef_Normalized=['Normalized Coef Price','Normalized Coef Generation forecast (FR)','Normalized Coef Load forecast (FR)']
#Title_Coef_Abs_Normalized=['Abs Normalized Coef Price','Abs Normalized Coef Generation forecast (FR)','Abs Normalized Coef Load forecast (FR)']
#Title_Coef_x_train=['Coef_X_train Price','Coef_X_train Generation forecast (FR)','Coef_X_train Load forecast (FR)']

#2015_BE
#Title_Coef=["Coef Price","Coef Generation forecast (FR)","Coef Load forecast (FR)","Coef Wind Forecast (BE)","Coef Solar Forecast (BE)","Coef Load forecast (BE)"]
#Title_Coef_Normalized=["Normalized Coef Price","Normalized Coef Generation forecast (FR)","Normalized Coef Load forecast (FR)","Normalized Coef Wind Forecast (BE)","Normalized Coef Solar Forecast (BE)","Normalized Coef Load forecast (BE)"]
#Title_Coef_Abs_Normalized=["Abs Normalized Coef Price","Abs Normalized Coef Generation forecast (FR)","Abs Normalized Coef Load forecast (FR)","Abs Normalized Coef Wind Forecast (BE)","Abs Normalized Coef Solar Forecast (BE)","Abs Normalized Coef Load forecast (BE)"]
#Title_Coef_x_train=["Coef_X_train Price","Coef_X_train Generation forecast (FR)","Coef_X_train Load forecast (FR)","Coef_X_train Wind Forecast (BE)","Coef_X_train Solar Forecast (BE)","Coef_X_train Load forecast (BE)"]

#2015_NP
#Title_Coef=["Coef Price","Coef Load forecast","Coef Wind forecast (DK)","Coef Wind forecast (EE)","Coef Wind forecast (LT)","Coef Wind forecast (SE)","Coef Wind forecast (LV)","Coef Total Wind","Coef Hydro reservoir"]
#Title_Coef_Normalized=["Normalized Coef Price","Normalized Coef Load forecast","Normalized Coef Wind forecast (DK)","Normalized Coef Wind forecast (EE)","Normalized Coef Wind forecast (LT)","Normalized Coef Wind forecast (SE)","Normalized Coef Wind forecast (LV)","Normalized Coef Total Wind","Normalized Coef Hydro reservoir"]
#Title_Coef_Abs_Normalized=["Abs Normalized Coef Price","Abs Normalized Coef Load forecast","Abs Normalized Coef Wind forecast (DK)","Abs Normalized Coef Wind forecast (EE)","Abs Normalized Coef Wind forecast (LT)","Abs Normalized Coef Wind forecast (SE)","Abs Normalized Coef Wind forecast (LV)","Abs Normalized Coef Total Wind","Abs Normalized Coef Hydro reservoir"]
#Title_Coef_x_train=["Coef_X_train Price","Coef_X_train Load forecast","Coef_X_train Wind forecast (DK)","Coef_X_train Wind forecast (EE)","Coef_X_train Wind forecast (LT)","Coef_X_train Wind forecast (SE)","Coef_X_train Wind forecast (LV)","Coef_X_train Total Wind","Coef_X_train Hydro reservoir"]

#2015_NP_no_wind1
#Title_Coef=['Coef Price','Coef Grid load forecast [MW]','Coef Wind power forecast (SE) [MWh]','Coef Total Wind [MWh]','Coef Hydro reservoir [GWh]']
#Title_Coef_Normalized=['Normalized Coef Price','Normalized Coef Grid load forecast [MW]','Normalized Coef Wind power forecast (SE) [MWh]','Normalized Coef Total Wind [MWh]','Normalized Coef Hydro reservoir [GWh]']
#Title_Coef_Abs_Normalized=['Abs Normalized Coef Price','Abs Normalized Coef Grid load forecast [MW]','Abs Normalized Coef Wind power forecast (SE) [MWh]','Abs Normalized Coef Total Wind [MWh]','Abs Normalized Coef Hydro reservoir [GWh]']
#Title_Coef_x_train=['Coef_X_train Price','Coef_X_train Grid load forecast [MW]','Coef_X_train Wind power forecast (SE) [MWh]','Coef_X_train Total Wind [MWh]','Coef_X_train Hydro reservoir [GWh]']

#Full_Dataset-NP
#Title_Coef=['Coef Price','Coef load forecast','Coef Wind forecast']
#Title_Coef_Normalized=['Normalized Coef Price','Normalized Coef load forecast','Normalized Coef Wind forecast']
#Title_Coef_Abs_Normalized=['Abs Normalized Coef Price','Abs Normalized Coef load forecast','Abs Normalized Coef Wind forecast']
#Title_Coef_x_train=['Coef_X_train Price','Coef_X_train load forecast','Coef_X_train Wind forecast']

#2015_NP_no_wind
Title_Coef=['Coef Price','Coef Grid load forecast [MW]','Coef Wind forecast (DK) [MWh]','Coef Wind forecast (SE) [MWh]','Coef Total Wind [MWh]','Coef Hydro reservoir [GWh]']
Title_Coef_Normalized=['Normalized Coef Price','Normalized Coef Grid load forecast [MW]','Normalized Coef Wind forecast (DK) [MWh]','Normalized Coef Wind forecast (SE) [MWh]','Normalized Coef Total Wind [MWh]','Normalized Coef Hydro reservoir [GWh]']
Title_Coef_Abs_Normalized=['Abs Normalized Coef Price','Abs Normalized Coef Grid load forecast [MW]','Abs Normalized Coef Wind forecast (DK) [MWh]','Abs Normalized Coef Wind forecast (SE) [MWh]','Abs Normalized Coef Total Wind [MWh]','Abs Normalized Coef Hydro reservoir [GWh]']
Title_Coef_x_train=['Coef_X_train Price','Coef_X_train Grid load forecast [MW]','Coef_X_train Wind forecast (DK) [MWh]','Coef_X_train Wind forecast (SE) [MWh]','Coef_X_train Total Wind [MWh]','Coef_X_train Hydro reservoir [GWh]']


#LEAR prediction
lear_list = [item for sublist in results['LEAR'] for item in sublist]

#Real Price
price_aux=df_test.Price.values

#coefficients
coef_aux=pd.DataFrame(results["coef"])
coef_aux.columns=Title_Coef

#normalized coefficients
coef_normalized_aux=(coef_aux/sum_abs)
coef_normalized_aux.columns=Title_Coef_Normalized

#absolute values of the normalized coefficients
coef_abs_normalized_aux=coef_normalized_aux.abs()
coef_abs_normalized_aux.columns=Title_Coef_Abs_Normalized

#product of Xtrain and Coefficients
coefXx_test_aux=pd.DataFrame(results['coefXx_test'])
coefXx_test_aux.columns=Title_Coef_x_train

final = pd.concat([coef_aux,coef_normalized_aux, coef_abs_normalized_aux, coefXx_test_aux], axis = 1)

final['Intercept']=results['intercept']
final['LEAR']=lear_list
final['Real Price']=price_aux
final['MAE']=np.abs(price_aux-np.array(lear_list))
cols = final.columns.tolist()
cols = cols[-3:] + cols[:-3]
final = final[cols]

final.insert(0, 'Dates', df_test.index)

final.to_csv('C:/Users/mmmap/epftoolbox/Final_Out/Results_2015_NP_no_wind_CW365_YT6.csv', index = False)


