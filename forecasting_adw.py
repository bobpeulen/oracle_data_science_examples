#%%writefile ./model_artifacts_v14_with_adw/score.py

## when changing forecast, only change below two AND change SQL query number of minutes
#set minutes for history and forecast here. 60 means that the last 60 minutes from the log files will be used to predict future 10 minutes.


import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pickle
import gzip
from prophet import Prophet
import ads
import os
import configparser
import shutil
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
import urllib
import re
import sqlalchemy
from sqlalchemy import create_engine
import cx_Oracle
from ocifs import OCIFileSystem
import cx_Oracle


def load_model():
    class DummyModel:
        def __init__(self):
            pass
    return DummyModel()

#create folder for input files
if not os.path.exists("input_files"):
    os.makedirs("input_files")

############################
############################

#variables to forecast
list_variables = ['EXTRACT_LAG', 'DATA_PUMP_READ_LAG', 'REPLICAT_READ_LAG', 'REPLICAT_APPLY_LAG', 'TOTAL_LAG']

number_of_historical_minutes_min1 = number_of_historical_minutes - 1


def predict(data, model=load_model()):
    
    number_of_historical_minutes = 2880
    forecast_in_minutes = 60  #10 minutes will be forecast
    
    #test print root
    print('Get current working directory : ', os.getcwd())
       
    input_files_location = os.getcwd() + "/input_files/"
    print("Input file full location " + input_files_location)

    #get the bucket name, namespace, and full file name
    file_name = data['file_name']
    bucket_name = data['bucket_name']
    namespace = data['namespace']
    
    #get full location in bucket
    full_location_in_bucket = "oci://" + bucket_name + "@" + namespace + "/"+file_name
    print("full location in bucket " + full_location_in_bucket)
       
    raw_input_from_zip = pd.read_csv(full_location_in_bucket, names=['SOURCE_HB_TS','EXTRACT', 'EXTRACT_LAG','DATA_PUMP','DATA_PUMP_READ_LAG','REPLICAT','REPLICAT_READ_LAG','REPLICAT_APPLY_LAG','TOTAL_LAG'], header=None)
    
    #########################################################
    ######################################################### Push raw .csv file to database
    #########################################################
    
    connection_parameters = {
        'user_name': 'OMLUSER',
        'password': '',
        'service_name': 'pocdb_high',
        'wallet_location': "/home/datascience/model-server/app/deployed_model/credentials/Wallet_pocdb.zip",
        }
    
    print("Log file loaded, now push to database staging table")
    ## push results to database
    raw_input_from_zip.ads.to_sql('maersk_stage_v3', connection_parameters=connection_parameters, if_exists="append")
    
    
    #########################################################
    ######################################################### Query last 60 minutes from database, staging table
    #########################################################
    
    ##### dpulicates?    
    
    ## WHERE MAERSK_STAGE_V3.SOURCE_HB_TS >= SYSDATE - INTERVAL '1' HOUR
    raw_input_from_zip = pd.DataFrame.ads.read_sql("SELECT * FROM (select * from MAERSK_STAGE_V3 ORDER BY SOURCE_HB_TS DESC) suppliers2 WHERE rownum <= 2880 ORDER BY rownum DESC", connection_parameters=connection_parameters)
    print("Shape of query to dataframe")
    print(raw_input_from_zip.shape)    
    print(raw_input_from_zip.head())
    
    #only 02 rows
    input_all_minutes = raw_input_from_zip[raw_input_from_zip['EXTRACT']=='EXTPRD02'] #
    
    #sort on time. So that latest 60 rows are actually latest 60 minutes. Convert to timestamp first.
    input_all_minutes['SOURCE_HB_TS'] = pd.to_datetime(input_all_minutes['SOURCE_HB_TS'])
    input_all_minutes.sort_values(by='SOURCE_HB_TS', ascending = True, inplace = True) 

    #drop duplicates. Many duplicate rows
    input_all_minutes.drop_duplicates(subset=['SOURCE_HB_TS'], keep='first', inplace=True)

    #get the latest 60 minues only.
    input_60_minutes = input_all_minutes.tail(number_of_historical_minutes)

    print("Shape after filter on EXTRACT and dropping duplicates")
    print(input_60_minutes.shape)
    
    #add random id for id set
    set_id = "set_id_"+ str(uuid.uuid4())
    

    #loop through the 5 variables. Build forecast for each one of them and push to database
    for variable in list_variables:
        
        print("-------------------------------------------------------")
        print("Start variable " + variable)
        print(type(variable))
        
        #create empty list
        list_to_db =  []
    
        #only select one variable to filter on
        df_input_1_variable = input_60_minutes[['SOURCE_HB_TS', variable]]
        
        #change name to what Prophet expects       
        df_input_1_variable.rename(columns={"SOURCE_HB_TS": "ds", variable:"y"}, inplace=True)
        
        #convert ds to timeframe
        df_input_1_variable['ds'] = pd.to_datetime(df_input_1_variable['ds'])
        
        #start date
        start_date = df_input_1_variable['ds'].min()
        print("Start time of " + variable + " is " + str(start_date))

        #end date
        end_date = df_input_1_variable['ds'].max()
        print("End time of " + variable + " is " + str(end_date))

        #general settings
        print(df_input_1_variable)
        m_1 = Prophet(changepoint_prior_scale=0.9)
        m_1.fit(df_input_1_variable)

        #create a empty dataframe with forecast dates
        future_1 = m_1.make_future_dataframe(periods=forecast_in_minutes, freq="min")
        print("Future 1 df " +str(future_1))

        #use the model to predict
        forecast_1 = m_1.predict(future_1)
        forecast_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        print("Full forecast + future " + str(forecast_1))
        
        print(forecast_1.shape)

        # average actual 60 minutes as input
        avg_input = forecast_1['yhat'].head(number_of_historical_minutes).mean()
        max_input = forecast_1['yhat'].head(number_of_historical_minutes).max()
        min_input = forecast_1['yhat'].head(number_of_historical_minutes).min()
        print("Average in input value is " + str(avg_input))
        print("Max input value " + str(max_input))
        print("Min input value " + str(min_input))
        print(" ------------------- ")

        #average 10 minutes as forecasted
        avg_forecast = forecast_1['yhat'].tail(forecast_in_minutes).mean()
        max_forecast = forecast_1['yhat'].tail(forecast_in_minutes).max() 
        min_forecast = forecast_1['yhat'].tail(forecast_in_minutes).min() 
        print("Average in forecast value is " + str(avg_forecast))
        print("Max forecast value " + str(max_forecast))
        print("Min forecast value " + str(min_forecast))
        print(" ------------------- ")

        #increase/decrease based on actual vs forecasted
        diff_perc_avg = round(((avg_forecast-avg_input)/avg_input)*100,2)
        diff_perc_max = round(((max_forecast-max_input)/max_input)*100,2)
        diff_perc_min = round(((min_forecast-min_input)/min_input)*100,2)
        print("Expected increase or decreases in the coming 10 minutes is " + str(diff_perc_avg)+str("%"))
        print("Expected increase/decrease in max "  + str(diff_perc_max)+str("%"))
        print("Expected increase/decrease in min "  + str(diff_perc_min)+str("%"))        
        
        #add to list
        list_to_db.append([set_id, variable, start_date, end_date, avg_input, max_input, min_input, avg_forecast, max_forecast, min_forecast, diff_perc_avg, diff_perc_max, diff_perc_min])
        df_to_db = pd.DataFrame(list_to_db, columns =['set_id', 'variable', 'start_date', 'end_date', 'avg_input', 'max_input', 'min_input','avg_forecast', 'max_forecast', 'min_forecast', 'diff_perc_avg', 'diff_perc_max', 'diff_perc_min'])

        #################
        ################ push results to adw
       
        

        
        ## push results to database
        df_to_db.ads.to_sql('maersk_logs_v7', connection_parameters=connection_parameters, if_exists="append")
                            
                            
        #delete list for next loop
        del list_to_db
                        
        ### create one line for both historical values and future values. Only for total lag
        if variable == 'TOTAL_LAG':
            df_full_picture = forecast_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            #loc_end = number_of_historical_minutes - 1 
            #add a 1 to last rows and 0 to other rows for color what is future and what is not.
            df_full_picture['value'] = 'future' #add column named value with only 'future as value'
            df_full_picture['value'].loc[0:number_of_historical_minutes_min1]= 'history'  #change first xx rows to 'history'
                        
            df_full_picture.ads.to_sql('maersk_one_line_v3', connection_parameters=connection_parameters, if_exists="replace")
            

            
        else:
            continue
        
        print()
        print("-----------------------")
        print("Table updated with results for " + variable)
        print("-----------------------")
        
        
    
    
    #return {'diff_perc_input_vs_forecast':diff_perc_input_vs_forecast, 'avg_input':avg_input, 'avg_forecast':avg_forecast}
