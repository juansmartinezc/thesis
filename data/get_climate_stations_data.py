import os
import pandas as pd
from dotenv import load_dotenv
from utils.states_codes import states_dict
from data.get_crop_yield_data import get_crop_yield, save_crop_yield_data
from data.get_climate_data import get_scan_stations_data, save_stations_data


'''
Estados con datos disponibles: ['AL', 'AR', 'AZ', 'CA', 'CO', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MD', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NJ', 'NM', 'NY', 'OH', 'OK', 'OR', 'PA', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY']

Estados sin datos disponibles: ['AS', 'CT', 'GU', 'HI', 'MA', 'ME', 'MP', 'NH', 'NV', 'PR', 'RI', 'VI', 'VT']

'''

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

# Obtener API Key desde variables de entorno
usda_api_key = os.environ.get("USDA_API_KEY")

df_stations = get_scan_stations_data()
save_stations_data(df_stations)


