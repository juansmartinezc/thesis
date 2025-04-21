import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataprep.eda import create_report
from eda.aux_functions import correlation_analysis, plot_monthly_crop_yield, plot_yearly_crop_yield, plot_crop_yield_by_status, plot_crop_yield_by_status_top_20

#############Cargamos variables de entorno#######################
load_dotenv()

#################################################################
######################Analisis EDA###############################
#################################################################

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
file_path = f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv'

df = pd.read_csv(file_path)
report_directory = 'reports'

#################Columnas a eliminar#############################

columns_to_drop = ['Unnamed: 0', 'state_alpha', 'county_code', 'stationTriplet',
       'stationId', 'name', 'lat_centroid', 'lon_centroid', 'latitude',
       'longitude', 'unit_desc']

results_directory = 'results'

#Eliminacion de columnas
df = df.drop(columns=columns_to_drop)
print(df.columns)

###############Creacion de directorio para almacenar reportes############################

os.makedirs(report_directory, exist_ok=True)


#############################Creacion de reportes#########################################

#report = create_report(df)
#report.save(f'{report_directory}/{report}')


##########################Almacenar el conjunto de entrenamiento###########################
df.to_csv(f'{report_directory}/input_data.csv', index=False)


###################Analisis numerico##############
numeric_summary = df.describe()
plt.rcParams['figure.figsize'] = (10, 6)

correlation_matrix = correlation_analysis(df)

#################################Visualización de rendimiento#############################

plot_monthly_crop_yield(df, results_directory)

# Evolución del rendimiento a lo largo del tiempo
df_grouped = df.groupby('year')['Value'].mean().reset_index()

plot_yearly_crop_yield(df_grouped, results_directory)

plot_crop_yield_by_status(df, results_directory = 'results')

plot_crop_yield_by_status_top_20(df, results_directory = 'results')

