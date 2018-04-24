#Objective: The Main objective of this program is to forecast the ticket volumes for each month 
                                # TIME SERIES ANALYSIS
#           The sub steps include:
            # 1. TAKING THE INCIDENT FILE AND PREPROCESSING THE DATA TO GET IN THE DESIRED FORMAT FOR TIME SERIES ANALYSIS
            # 2. CHECKING FOR THE SEASONALITY, TREND AND RESIDUAL ANALYSIS
            # 3. FINDING THE BEST CONGIURATION PARAMETER REQUIRED FOR TIME SERIES ANALYSIS USING ARIMA
            # 4. FINDING THE BEST CONGIURATION PARAMETER REQUIRED FOR TIME SERIES ANALYSIS USING EXPONENTIAL SMOOTHING 
            # 5. SAVING THE FORECASTED VALUES IN THE PATH MENTIONED THROUGH COMMAND LINE
            # 6. NEXT TIME WHEN THE JOB IS RUN AGAIN THE DATA IS MODELED AND FORECASTED AGAIN FOR NEXT 3 MONTHS

#Import the required libraries
import pandas as pd
import numpy as np
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
import warnings
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import commands
import logging
import re
from optparse import OptionParser
import argparse
import datetime
from datetime import date
import calendar
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

class arima_model():

    # create a differenced series
    def difference(self, data, interval=1):
        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]

    # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
    def evaluate_arima_model(self, X, arima_order):
        # prepare training dataset
        X = X.astype('float32')
        train_size = int(len(X) * 0.50)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
    
        # make predictions
        predictions = list()
        for t in range(len(test)):
            # difference data
            months_in_year = 12
            diff = self.difference(history, months_in_year)
            model = ARIMA(diff, order=arima_order)
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = self.inverse_difference(history, yhat, months_in_year)
            predictions.append(yhat)
            history.append(test[t])
            
            # calculate out of sample error
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mse)
        return rmse

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(self, dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self.evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s RMSE=%.3f' % (order,mse))
                    except:
                        continue
        print('Best ARIMA %s RMSE=%.3f' % (best_cfg, best_score))
        return best_cfg


    #Model building using train data and validating against the test data. Returning the rmse for arima
    def arima_validation(self, series, best_cfg):
        try:
            split_point = len(series) - 12
            train, test = series[0:split_point], series[split_point:]
            history = [x for x in train]

            # make predictions
            arima_preds = list()
            for t in range(len(test)):
                # difference data
                months_in_year = 12
                diff = self.difference(history, months_in_year)
                model = ARIMA(diff, best_cfg)
                model_fit = model.fit(trend='nc', disp=0)
                yhat = model_fit.forecast()[0]
                yhat = self.inverse_difference(history, yhat, months_in_year)
                arima_preds.append(yhat)
                history.append(test[t])

            arima_preds = [int(x) for x in arima_preds]
            #calculating the mean squared error
            mse = mean_squared_error(test, arima_preds)
            rmse = sqrt(mse)
            msg = 'The arima data is validated and rmse is calculated'

        except Exception as e:
            msg = e

        return rmse, arima_preds, msg

    #Making the final prediction for arima using the best configuration parameter
    def arima_final_prediction(self,data, best_cfg):
        try:
            history = [x for x in data]
            months_in_year = 12

            #Difference the data
            diff = self.difference(history, months_in_year)
            model = ARIMA(diff, order = best_cfg)
            model_fit = model.fit(trend='nc',disp = 0)
            yhat = model_fit.forecast(steps=3)[0]

            predictions = list()
            for i in range(len(yhat)):
                yhat_orig = self.inverse_difference(history, yhat[i], months_in_year)
                predictions.append(yhat_orig)
                history.append(yhat_orig)
            
            predictions = [int(x) for x in predictions]
            msg = 'Forecast for next 3 months using ARIMA Model is %s' % (predictions)

        except Exception as e:
            msg = e
        
        return predictions, msg


################################################################################################################################################
#                               CREATING THE CLASS OBJECTS FOR STORING THE FORECASTED FILE
################################################################################################################################################

class forecast_file_preprocessing():

    def last_day_of_month(self,any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4) 
        return next_month - datetime.timedelta(days=next_month.day)

    def filter_by_group(self, df, dept_group):
        try:
            df = df[df['Customer'].notnull()]
            df['Customer'] = df['Customer'].map(lambda x:x.lower())
            dept_group = dept_group.lower()
            df = df.loc[df['Customer'] == dept_group, 'Opened_at']
            df = pd.DataFrame(df)
            df = df.reset_index(drop = True)
            msg = "The data is filtered by the group {} after doing some preprocessing".format(dept_group.upper())
        except Exception as e:
            msg = e
            print(msg)
        return df, msg


    def preprocessing_volume(self, incident_df):
    
        try:
            #converting the object data type to date time 
            incident_df['Opened_at'] = pd.to_datetime(incident_df['Opened_at'], format="%Y-%m-%d")

            data = incident_df['Opened_at'].groupby(incident_df.Opened_at.dt.to_period("M")).agg('count')
            data.index = data.index.to_timestamp()

            # get the min date and max date and fill the missing dates with zero value
            min_date = min(data.index)
            max_date = max(data.index)
            idx = pd.date_range(min_date, max_date)
            data.index = pd.DatetimeIndex(data.index)
            data = data.reindex(idx, fill_value=0)

            # re-index the data and resample it using Month
            data = data.reindex(data.index).resample('M').sum()
            msg = "The data is preprocessed into a series form to pass it to time series models"
    
        except Exception as e:
            msg = e
            print(msg)
            exit(1)

        return data, msg

    def filter_by_cust_dept(self, df, cust, dept_group):
        try:
            #df = pd.read_csv('Customer_Data_Department.csv', parse_dates=['Opened_at'], dayfirst=True)
            df = df.loc[(df['Customer'] == cust) & (df['Department'] == dept_group)]
            df = df['Opened_at']
            df = pd.DataFrame(df)
            df = df.reset_index(drop = True)
            msg = "The data is filtered by the customer group {} and department group {} after doing some preprocessing".format(cust.upper(), dept_group.upper())
        except Exception as e:
            msg = e
            print(msg)
        return df, msg

    def filter_by_cust_dept_pty(self, df, cust, dept_group, pty):
        try:
            #df = pd.read_csv('Customer_Data_Department.csv', parse_dates=['Opened_at'], dayfirst=True)
            df = df.loc[(df['Customer'] == cust) & (df['Department'] == dept_group) & (df['Priority'] == pty)]
            df = df['Opened_at']
            df = pd.DataFrame(df)
            df = df.reset_index(drop = True)
            msg = "The data is filtered by the customer group {}, department group {} , priority {}".format(cust.upper(), dept_group.upper(), pty.upper())
        except Exception as e:
            msg = e
            print(msg)
        return df, msg

    #Redaing the input file from the path specified
    def input_file_read(self, FILENAME, filetype):
        if filetype == 'csv':
            inc_df = pd.read_csv(FILENAME)
        elif filetype == 'xls':
            inc_df = pd.read_excel(FILENAME)
        return inc_df


    def resoulution_time_calc(self, df, cust, dept, pty):
        try:
            df = df = df.loc[(df['Customer'] == cust) & (df['Department'] == dept) & (df['Priority'] == pty)]
            rtime = df['Resolution_Time'].max()
            rtime = round(rtime, 2)
            msg = 'The max Resolution Time {} for this group {}, {}, {} is '.format(rtime, cust, dept, pty)
        except Exception as e:
            msg = e
            print(msg)
        return rtime, msg

    def calc_resources(self, fc, rtime):
        num_resources = []
        for i in range(len(fc)):
            num_days = fc[i]*rtime
            num_resources.append(round(num_days/(20)))
        return num_resources

    def res_time_column_check(self, df):
        try:
            if 'Resolution_Time' not in df.columns:
                df['Resolution_Time'] = df['Closed_at'] - df['Opened_at']
                df['Resolution_Time'] = df['Resolution_Time'].map(lambda x: x.days)
                df['Resolution_Time'] = df['Resolution_Time'].map(lambda x: 0.5 if x==0 else x)
                msg = 'The column Resolution_Time is created successfully for this data'
        except Exception as e:
            msg = e
        return df, msg
