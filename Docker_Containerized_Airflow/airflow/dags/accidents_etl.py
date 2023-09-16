from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime,timedelta

from Milestone3_1 import clean_and_transform
from Milestone3_2 import augment_dataset
from DashboardCreator import createDashboard

import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine


source = '2011_Accidents_UK.csv'
external_sources = ['/opt/airflow/data/LSOA_data_England.csv','/opt/airflow/data/LSOA_data_Wales.csv']

look_up_cleaned_transformed = 'look_up_cleaned_transformed.csv'
look_up_table_name = 'look_up_table.csv'

cleaned_transformed_name = 'cleaned_transformed.csv'
processed_data_name = 'processed_accidents_2011.csv'




def load_to_postgres(filename,look_up_table): 
    df = pd.read_csv(filename)
    df_lookup = pd.read_csv(look_up_table)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/accidents_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    try:
        df.to_sql(name = 'UK_Accidents_2011',con = engine,if_exists='fail')
    except ValueError:
        print("Accidents table already exists.")
    try:
        df_lookup.to_sql(name = 'lookup_table',con = engine,if_exists='fail')
    except ValueError:
        print(" Look up table already exists.")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'accidents_etl_pipeline',
    default_args=default_args,
    description='accidents etl pipeline',
)
with DAG(
    dag_id = 'accidents_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['accidents-pipeline'],
)as dag:
    extract_clean_transform_dataset_task= PythonOperator(
        task_id = 'extract_clean_transform_dataset_task',
        python_callable = clean_and_transform,
        op_kwargs={
            "file_name": '/opt/airflow/data/'+str(source),
            "df_final_name": '/opt/airflow/data/'+str(cleaned_transformed_name),
            "look_up_table_name": '/opt/airflow/data/'+str(look_up_cleaned_transformed)
        },
    )
    augment_dataset_task= PythonOperator(
        task_id = 'augment_dataset_task',
        python_callable = augment_dataset,
        op_kwargs={
            "sourcelist": external_sources,
            "look_up_table": '/opt/airflow/data/'+str(look_up_cleaned_transformed),
            "cleaned_transformed" : '/opt/airflow/data/'+str(cleaned_transformed_name),
            "look_up_final": '/opt/airflow/data/'+str(look_up_table_name),
            "augmented_data" : '/opt/airflow/data/'+str(processed_data_name),
            
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename":'/opt/airflow/data/'+str(processed_data_name),
            "look_up_table": '/opt/airflow/data/'+str(look_up_table_name)
        },
    ),
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = createDashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/"+str(processed_data_name),
            'look_up_table_filename' :'/opt/airflow/data/'+str(look_up_table_name)
        },
    )
    


    
    extract_clean_transform_dataset_task >> augment_dataset_task >> load_to_postgres_task >> create_dashboard_task

    
    



