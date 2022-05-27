   
"""
Classes and functions to implement the LEAR model for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

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

from epftoolbox.models._utils import ScalingTypes, FeatureLags


class LEAR(object):
    """
    New version of the class used to build a LEAR model, recalibrate it, and predict DA electricity prices.
    New features:
    - User can now supply the lags (in days) used for creating new lagged exogenous features SEPARATELY.
      Previously all exog. features used d-1 and d-2. NOTE: the original documentation wrongly stated d-1 and d-7!
    - When the original X features get split/reshaped into 24 hourly features, the hourly features now get checked
      for being constant. Constant features get removed, and their original index recorded.
      This is needed because during calibration all X features get scaled by MAD or std, and
      if the scaling factor is 0, it leads to an error. This error is very common for solar data (eg. solar power
      is always 0 at midnight)!
    - Order of the newly built hourly and lagged features is changed to be grouped by feature:lag:hour, instead of
      hour:lag:feature. This makes it easier to track the coefficients of the different features when analyzing the
      results. This is less efficient wrt memory when building the array, but the practical difference is marginal.
    - Not recalibrating in every forecast loop (daily) is now more convenient. Previously all the methods were there,
      but had to be called manually in the forecast loop. In addition training data arrays would be constructed, even
      though only test data is needed.
    - Changed variable names to Python naming conventions (eg. predDatesTrain -> pred_dates_train)
    - Added parameter for choosing the scaling type for pre-processing X. The methods for choosing scaling were there,
      but the used sclaing was hard-coded.
    An example on how to use this class is provided :ref:`here<learex2>`.
    Parameters
    ----------
    calibration_window : int, optional
        Calibration window (in days) for the LEAR model.
    lags: list[int] or list[list[int]], optional
        The number of lags for the exogenous features. If list of ints, then the same
        lags are applied for all exogenous features. If list of lists of ints then
        the number of lists must match the number of exogenous features.
    scaling_type : str, optional
        Must be one of:
        - ``'Norm'`` for normalizing the data to the interval [0, 1].
        - ``'Norm1'`` for normalizing the data to the interval [-1, 1].
        - ``'Std'`` for standarizing the data to follow a normal distribution.
        - ``'Median'`` for normalizing the data based on the median as defined in as defined in `here
            <https://doi.org/10.1109/TPWRS.2017.2734563>`_.
        - ``'Invariant'`` for scaling the data based on the asinh transformation (a variance stabilizing
            transformations) as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.
    """

    def __init__(self, calibration_window=364 * 3, lags=[1, 2], scaling_type="Invariant"):

        # Calibration window in hours
        self.calibration_window = calibration_window
        self.lags = FeatureLags(lags)
        if not ScalingTypes.has_value(scaling_type):
            raise ValueError("Invalid scaling type specified, check documentation!")
        self.scaling_type = ScalingTypes(scaling_type)
        self.const_features = None
        self.scaler_X = None
        self.scaler_Y = None
    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, X_train, Y_train):
        """Function to recalibrate the LEAR model. 
        
        It uses a training (X_train, Y_train) pair for recalibration
        
        Parameters
        ----------
        X_train : numpy.array
            Input in training dataset. It should be of size *[n,m]* where *n* is the number of days
            in the training dataset and *m* the number of input features
        
        Y_train : numpy.array
            Output in training dataset. It should be of size *[n,24]* where *n* is the number of days 
            in the training dataset and 24 are the 24 prices of each day
                
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model        
        
        """

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Y_train], self.scaler_Y = scaling([Y_train], self.scaling_type.value)

        # # Rescaling all inputs except dummies (7 last features)
        [X_train_no_dummies], self.scaler_X = scaling([X_train[:, :-7]], self.scaling_type.value)
        X_train[:, :-7] = X_train_no_dummies
        #auux=[]

        self.models = {}
        for h in range(24):
            # Estimating lambda hyperparameter using LARS
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)
            param = param_model.fit(X_train, Y_train[:, h]).alpha_
            # Re-calibrating LEAR using standard LASSO estimation technique
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(X_train, Y_train[:, h])

            self.models[h] = model
            
    def predict(self, X):
        """Function that makes a prediction using some given inputs.
        
        Parameters
        ----------
        X : numpy.array
            Input of the model.
        
        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros(24)
        Inter= np.zeros(24)

        # # Rescaling all inputs except dummies (7 last features)
        X_no_dummies = self.scaler_X.transform(X[:, :-7])
        X[:, :-7] = X_no_dummies

        # Predicting the current date using a recalibrated LEAR
        for h in range(24):
            # Predicting test dataset and saving
            Yp[h] = self.models[h].predict(X)
            #Inter[h]=self.models[h].intercept_
        Yp = self.scaler_Y.inverse_transform(Yp.reshape(1, -1))
        #Inter = self.scaler_Y.inverse_transform(Inter.reshape(1, -1))

        return Yp

    def recalibrate_predict(self, X_train, Y_train, X_test):
        """Function that first recalibrates the LEAR model and then makes a prediction.
        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.
        
        Parameters
        ----------
        X_train : numpy.array
            Input of the training dataset.
        X_test : numpy.array
            Input of the test dataset.
        Y_train : numpy.array
            Output of the training dataset.
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """

        self.recalibrate(X_train=X_train, Y_train=Y_train)

        Yp = self.predict(X=X_test)

        return Yp

    def _build_and_split_XYs(self, df_train, df_test=None, date_test=None, calibration=True):
        """
        Internal function that generates the X,Y arrays for training and testing based on pandas dataframes
        Parameters
        ----------
        df_train : pandas.DataFrame
            Pandas dataframe containing the training data
        df_test : pandas.DataFrame
            Pandas dataframe containing the test data
        date_test : datetime, optional
            If given, then the test dataset is only built for that date
        calibration : bool, optional
            If True, only test array is built
        Returns
        -------
        list[numpy.array]
            [X_train, Y_train, X_test] as the list containing the (X,Y) input/output pairs for training,
            and the input for testing
        """

        # Checking that the first index in the dataframes corresponds with the hour 00:00
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        if n_exogenous_inputs > 0:
            # If list of ints given repeat it for all exog. features
            if isinstance(self.lags[0], int):
                self.lags.expand_lags(n_exogenous_inputs)

            # If list of ints given repeat it for all exog. features
            elif isinstance(self.lags[0], list):
                if len(self.lags) != n_exogenous_inputs:
                    msg = "there are {} exog. features but {} set of lags \
                          is defined".format(n_exogenous_inputs,len(self.lags))
                    raise ValueError(msg)

            # Count how many features there will be in total
            n_new_exog_feat = 0
            for feat_lags in self.lags:
                n_new_exog_feat += (len(feat_lags) + 1) * 24
            n_features = 96 + 7 + n_new_exog_feat
        else:
            # no exog. features
            n_new_exog_feat = 0
            n_features = 96 + 7 + n_new_exog_feat


        # Extracting the predicted dates for testing and training. Some data needs to be left out,
        # as it does not have enough history for the specified lags

        # We define the potential time indexes that have to be forecasted in training
        # and testing
        # 7 comes from D-7 for historical prices
        if n_exogenous_inputs > 0:
            max_lag=max(self.lags.get_max_lag(), 7)
        else:
            max_lag = 7
        if calibration:
            index_train_long = self._crop_to_lags(df_train,max_lag).index
        # For testing, the test dataset is different depending on whether a specific test
        # dataset is provided
        if date_test is None:
            index_test_long = self._crop_to_lags(df_test, max_lag).index
        else:
            # I have no idea why this is using 23 hours so I didnt refactor yet
            index_test_long = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        # Creating two auxiliary dataframes to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day

        # These dataframes hold hourly indices, organized in days x hours wide format
        if calibration:
            index_train_wide = self._hourly_long_to_wide_index(index_train_long)
        index_test_wide = self._hourly_long_to_wide_index(index_test_long)

        # Preallocating in memory the X and Y arrays
        if calibration:
            X_train = np.zeros([index_train_wide.shape[0], n_features])
            Y_train = np.zeros([index_train_wide.shape[0], 24])
        else:
            X_train = None
            Y_train = None
        X_test = np.zeros([index_test_wide.shape[0], n_features])

        # Variable keeping track of nr of features built so far
        feature_index = 0

        # Adding the historical prices during days D-1, D-2, D-3, and D-7
        # For each possible past day where prices can be included
        for lag in [1, 2, 3, 7]:
            # For each hour of a day
            for hour in range(24):
                # We define the corresponding past time indexs using the auxiliary dataframses
                if calibration:
                    past_index_train = self._get_lagged_index_at_hour(index_train_wide, hour, lag)
                past_index_test = self._get_lagged_index_at_hour(index_test_wide, hour, lag)

                # We include the historical prices at day D-past_day and hour "h"
                if calibration:
                    X_train[:, feature_index] = df_train.loc[past_index_train, 'Price']
                X_test[:, feature_index] = df_test.loc[past_index_test, 'Price']
                feature_index += 1

        # Adding the lagged version of the exogenous inputs
        # For each of the exogenous input
        for exog in range(n_exogenous_inputs):
            # For each possible past day where exogenous inputs can be included
            # lag 0 is the original feature!
            feature_lags = [0] + self.lags[exog]
            for lag in feature_lags:
                # For each hour of a day
                for hour in range(24):
                    # Defining the corresponding past time indices using the auxiliary dataframes
                    if calibration:
                        past_index_train = self._get_lagged_index_at_hour(index_train_wide, hour, lag)
                    past_index_test = self._get_lagged_index_at_hour(index_test_wide, hour, lag)

                    # Including the exogenous input at day D-past_day and hour "h"
                    if calibration:
                        X_train[:, feature_index] = df_train.loc[past_index_train, 'Exogenous ' + str(exog + 1)]
                    X_test[:, feature_index] = df_test.loc[past_index_test, 'Exogenous ' + str(exog + 1)]
                    feature_index += 1

        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        # For each day of the week
        for dayofweek in range(7):
            if calibration:
                X_train[index_train_wide.index.dayofweek == dayofweek, feature_index] = 1
            X_test[index_test_wide.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1
        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            if calibration:
                future_index_train = pd.to_datetime(index_train_wide.loc[:, 'h' + str(hour)].values)
                # Extracting Y value based on time indices
                Y_train[:, hour] = df_train.loc[future_index_train, 'Price']

        # Remove features with 0 scaling factor
        # Only check features for being constant when calibrating
        if calibration:
            if self.scaling_type in [ScalingTypes.NORM, ScalingTypes.NORM1, ScalingTypes.MEDIAN]:
                raise NotImplementedError("Checking scalability not yet implemented for all scaling types!")
            self._find_constant_features(X_train, self.scaling_type)
            X_train = self._remove_constant_features(X_train)
        X_test = self._remove_constant_features(X_test)

        return X_train, Y_train, X_test

    def _crop_to_lags(self, df, max_lag=7):
        """
        When lagged features are used, some data with not enough history needs to
        be discarded.
        Parameters
        ----------
        df: pandas.DataFrame
            the dataframe holding the data
        max_lag: int
            the largest lag applied to any feature in days
        Returns
        -------
        pandas.DataFrame
            cropped view of the pandas dataframe
        """
        return df.loc[df.index[0] + pd.Timedelta(days=max_lag):]

    def _hourly_long_to_wide_index(self,index_hourly_long):
        """
        Reshapes 1D hourly DatetimeIndex to 2D day x hour format.
        Parameters
        ----------
        index_hourly_long: pandas.core.indexes.datetimes.DatetimeIndex
            the original 1D index in hourly frequency
        Returns
        -------
        pandas.DataFrame
            the new 2D index in hourly frequency
        """
        index_daily=self._hourly_to_daily_index(index_hourly_long)
        index_hourly_wide = pd.DataFrame(index=index_daily, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            index_hourly_wide.loc[:, 'h' + str(hour)] = index_hourly_wide.index + pd.Timedelta(hours=hour)
        return index_hourly_wide

    def _hourly_to_daily_index(self, index_hourly):
        """
        Resamples DatetimeIndex from hourly frequency to daily frequency.
        Parameters
        ----------
        index: pandas.core.indexes.datetimes.DatetimeIndex
            the original index in hourly frequency
        Returns
        -------
        pandas.core.indexes.datetimes.DatetimeIndex
            the new index in daily frequency
        """
        assert(isinstance(index_hourly,pd.core.indexes.datetimes.DatetimeIndex))
        return index_hourly.round('1H')[::24]

    def _get_lagged_index_at_hour(self,index_hourly_wide, hour, lag):
        """
        Selects the appropriately lagged hourly index values from the original
        2D (day x hour) index dataframe's given hour column. This is needed
        as the lagged features get built hour-by-hour.
        Parameters
        ----------
        index_hourly_wide: pandas.DataFrame
            the original wide format 2D index (day x hour)
        Returns
        -------
        pandas.core.indexes.datetimes.DatetimeIndex
            the new 1D lagged index at a given hour
        """
        return pd.to_datetime(index_hourly_wide.loc[:, 'h' + str(hour)].values) - \
                                       pd.Timedelta(hours=24 * lag)

    def _find_constant_features(self, array, scaling_type=ScalingTypes.INVARIANT):
        if scaling_type is ScalingTypes.INVARIANT:
            self.const_features = np.where(mad(array[:, :-7], axis=0) == 0)[0]
        elif scaling_type is ScalingTypes.STD:
            self.const_features = np.where(np.std(array[:, :-7], axis=0) == 0)[0]
        else:
            raise ValueError("Invalid scaling type: " + scaling_type)
        return

    def _remove_constant_features(self, array):
        array = np.concatenate([np.delete(array[:, :-7], self.const_features, axis=1), array[:, -7:]], axis=1)
        return array

    def recalibrate_and_forecast_next_day(self, df, next_day_date):
        """Easy-to-use interface for daily recalibration and forecasting of the LEAR model.
        
        The function receives a pandas dataframe and a date. Usually, the data should
        correspond with the date of the next-day when using for daily recalibration.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and *N* exogenous inputs. 
            The index of the dataframe should be dates with hourly frequency. The columns 
            should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.
        next_day_date : datetime
            Date of the day-ahead.
        
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices.
        """

        # We define the new training dataset and test datasets 
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features. 
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X,Y pairs for predicting prices
        X_train, Y_train, X_test, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.recalibrate_predict(X_train=X_train, Y_train=Y_train, X_test=X_test)

        return Yp

    def forecast_next_day(self, df, next_day_date):
        
        """
        Easy-to-use interface forecasting with a pre-calibrated LEAR model.
        To be used when the model is not recalibrated every day.
        The function receives a pandas dataframe and a date.
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and *N* exogenous inputs.
            The index of the dataframe should be dates with hourly frequency. The columns
            should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.
        calibration_window : int
            Calibration window (in days) for the LEAR model.
        next_day_date : datetime
            Date of the day-ahead.
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices.
        """

        # We define the new training dataset and test datasets
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features.
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X,Y pairs for predicting prices
        _, _, X_test, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date, calibration=False)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.predict(X=X_test)

        return Yp
    
    def x_maninha(self, df,next_day_date):
        
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features. 
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X,Y pairs for predicting prices
        X_train, Y_train, X_test, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date)
        
        return X_test, Y_train

def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('.', 'datasets'),
                                  path_recalibration_folder=os.path.join('.', 'experimental_files'),
                                  dataset='PJM', years_test=2, calibration_window=364 * 3,
                                  begin_test_date=None, end_test_date=None):
    """Function for easy evaluation of the LEAR model in a test dataset using daily recalibration.
    The test dataset is defined by a market name and the test dates dates. The function
    generates the test and training datasets, and evaluates a LEAR model considering daily recalibration.
    An example on how to use this function is provided :ref:`here<learex1>`.
    Parameters
    ----------
    path_datasets_folder : str, optional
        path where the datasets are stored or, if they do not exist yet,
        the path where the datasets are to be stored.
    path_recalibration_folder : str, optional
        path to save the files of the experiment dataset.
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets,
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if
        the arguments ``begin_test_date`` and ``end_test_date`` are not provided.
    calibration_window : int, optional
        Number of days used in the training dataset for recalibration.
    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``end_test_date``. If either of them is not provided, the test dataset is built using the
        ``years_test`` argument. ``begin_test_date`` should either be a string with the following
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.
    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``begin_test_date``. If either of them is not provided, the test dataset is built using the
        ``years_test`` argument. ``end_test_date`` should either be a string with the following
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.
    Returns
    -------
    pandas.DataFrame
        A dataframe with all the predictions in the test dataset. The dataframe is also written to path_recalibration_folder.
    """

    # Checking if provided directory for recalibration exists and if not create it
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

    # For loop over the recalibration dates
    for date in forecast_dates:
        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp, Inter = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                     calibration_window=calibration_window)
        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_path)

    return forecast
