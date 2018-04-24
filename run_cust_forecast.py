#Objective: The Main objective of this program is to forecast the ticket volumes for each month 
								# TIME SERIES ANALYSIS
#			The sub steps include:
			# 1. TAKING THE INCIDENT FILE AND PREPROCESSING THE DATA TO GET IN THE DESIRED FORMAT FOR TIME SERIES ANALYSIS
			# 2. CHECKING FOR THE SEASONALITY, TREND AND RESIDUAL ANALYSIS
			# 3. FINDING THE BEST CONGIURATION PARAMETER REQUIRED FOR TIME SERIES ANALYSIS USING ARIMA
			# 4. SAVING THE FORECASTED VALUES IN THE PATH MENTIONED THROUGH COMMAND LINE
			# 5. THE FILE HAS TO STORED AS JSON WHICH CAN BE USED FOR REPORTING PURPOSE

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
import logging
import re
from math import sqrt
import datetime
from datetime import date
import calendar
from dateutil.relativedelta import relativedelta

#Importing the arima class object
from models_volume_forecast import forecast_file_preprocessing
from models_volume_forecast import arima_model 

#Importing the encapsulation mapping layer
from semantic_mapping_app import semantic_columns

#creating the arima class object
arima = arima_model()
fp   = forecast_file_preprocessing()


def customer_volume(APP_TYPE):
	
	#Generating the log file
	logger = logging.getLogger(__name__)
	logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='inc_vol_cust_forecast_log.log',
                    filemode='w')

	INPUT_FILE_PATH = 'Customer_Data_Department_AMPM.xls'
                            #==============================================================================================================#
                            #		TREND AND VOLUME FORECAST BASED ON CLIENT FOR THE INCIDENT DATA USING ARIMA 	    				   #
                            #==============================================================================================================#

	if INPUT_FILE_PATH is not None:
		filetype = INPUT_FILE_PATH.split('.')  #splitting the file name by '.' to get the file type
		file_ext = filetype[-1]

	#Getting the original column names and the renamed column names as per application from semantic mapping layer
	orig_cols, rename_columns = semantic_columns(APP_TYPE)

	#Reading the incident file provided 
	try:
		incident_df = fp.input_file_read(INPUT_FILE_PATH, file_ext)
		incident_df = incident_df[orig_cols]

	except Exception as e:
		logger.debug(e)
		print(e)
		exit(1)

	# Extracting the respective column names based on application
	try:
		incident_df.columns = rename_columns
		incident_df = incident_df[['Opened_at','Customer']]

	except Exception as e:
		logger.debug(e)
		print(e)
		logger.debug("There is an issue for taking the column names as per mapping of CLIENT TYPE")
		logger.debug(incident_df.columns)
		print("There is an issue for taking the column names as per mapping of CLIENT TYPE")
		print(incident_df.columns)
		exit(1)

							# PREPROCESSING OF THE INCIDENT DATA INTO A SERIES WHICH CAN BE PASSED TO THE TIME-SERIES MODEL
							# GETTING THE UNIQUE CUSTOMER VALUES

	cust_list = list(incident_df['Customer'].value_counts().index)

							#==============================================================================================================#
               	 # LOOPING THROUGH EACH CUSTOMER, APPLY PRE PROCESSING, PASSING TO MODEL, FORECASTING FOR NEXT 3 MONTHS, STORING AS CSV FILE    #
                            #==============================================================================================================#
	for cust in cust_list:

		try:
			#calling the filter by group function to filter data by group
			df_cust, msg = fp.filter_by_group(incident_df, cust)
			logger.info(msg)
			print(msg)

			#calling the pre-processing function
			data , msg  = fp.preprocessing_volume(df_cust)
			logger.info(msg)
			print(msg)

		################################################################################################################################################
		#												PASSING THE DATA TO THE ARIMA MODEL AND FINDING THE BEST CFG
		################################################################################################################################################
	
			# evaluate parameters
			p_values = range(0, 3)
			d_values = range(0, 3)
			q_values = range(0, 3)
			warnings.filterwarnings("ignore")
			print("processing for best cfg.....")
			best_cfg = arima.evaluate_models(data.values, p_values, d_values, q_values)
			print(best_cfg)

			##Extracting the arima rmse, arima predictions on test data
			arima_rmse, arima_preds, msg = arima.arima_validation(data, best_cfg)
			print(arima_preds)
			logger.info(msg)

			msg = 'RMSE for ARIMA model is %.3f' %arima_rmse
			logger.debug(msg)
			print(msg)

			#Extracting the arima predictions
			arima_forecast, msg = arima.arima_final_prediction(data, best_cfg)
			logger.debug(msg)
			print(msg)

		################################################################################################################################################
		#								SAVING THE RESULTS OF THE FORECAST TO THE DATABASE AS PER THE PATH MENTIONED IN COMMAND LINE
		################################################################################################################################################
	
			#Extracting the last index date and using that date to get the next 3 months date
			last_date = data.index[-1]
			last_date = pd.Timestamp(last_date)

			#Forecasting the next 3 months using the data for each customer
			forecast_months = []	
			for i in range(1, 4):
				forecast_months.append(fp.last_day_of_month(last_date + relativedelta(months=i)))	

			#This step is convert actual data to pandas dataframe with columns Customer, Month, Actuals and Forecast.
			data_df = pd.DataFrame(data)
			data_df['Month'] =data.index
			data_df = data_df.reset_index(drop=True)
			data_df.columns = ['Actuals','Month']
			data_df['Customer'] = cust
			data_df = data_df[['Customer','Month','Actuals']]
			length = len(data_df['Customer'])
			data_df['Forecast'] = pd.Series(None, index=data_df.index)

			#To the created dataframe of actual data, append the forecasted data also with customer name, forecasted month and forecasted actuals.
			for i in range(len(arima_forecast)):
				data_df.loc[len(data_df)] = [cust,forecast_months[i],None,arima_forecast[i]]

			#first time when loop runs, checking if the variable exists in locals and if not create the dataframe and append the forecasted data of each customer
			if 'forecast_df' not in locals():
				forecast_df =  pd.DataFrame(columns=['Customer','Month','Actuals','Forecast'])
				forecast_df = forecast_df.append(data_df, ignore_index=True)

			#In the else part it keeps adding each customer actual data along with the forecasted data to forecast_df dataframe
			else:
				forecast_df = forecast_df.append(data_df, ignore_index=True)

		except:
			msg = "The customer group {} has not been processed and this has been skipped for forecasting".format(cust.upper())
			logger.debug(msg)
			print(msg)
			continue

	#SAVING THE FORECASTED DATA TO THE CSV FILE IN THE DETSINATION PATH MENTIONED
	forecast_df['Month'] = forecast_df['Month'].dt.strftime('%Y-%m-%d')
	cust_forecast = df.to_json(orient='records', date_format = 'iso')[1:-1].replace('},{', '} {')
	return cust_forecast