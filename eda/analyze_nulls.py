import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from data.get_nasa import get_climate_missing_values, save_climate_missing_values


load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

######## Analyze null data ##########
df = pd.read_csv(f"{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv")
climate_missing_values = get_climate_missing_values(df)
null_values = df.isna().sum()
print(null_values)
save_climate_missing_values(climate_missing_values, file_name = 'historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv')
null_values = df.isna().sum()

