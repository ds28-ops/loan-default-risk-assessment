from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'monitor_and_retrain',
    default_args=default_args,
    description='Convert .txt to .csv and retrain model',
    schedule_interval='@hourly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

convert_task = BashOperator(
    task_id='convert_txt_to_csv',
    bash_command='python /opt/airflow/dags/scripts/txt_to_csv.py',
    dag=dag,
)

retrain_task = BashOperator(
    task_id='retrain_model',
    bash_command='python /opt/airflow/dags/scripts/retrain_model.py',
    dag=dag,
)

convert_task >> retrain_task
