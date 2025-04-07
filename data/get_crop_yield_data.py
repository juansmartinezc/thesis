import os
import pandas as pd
from dotenv import load_dotenv
from utils.states_codes import states_dict
from data.get_usda_data import get_usda_quick_stats


load_dotenv()
# Iterar sobre los estados y obtener los datos de rendimiento

usda_api_key = os.environ.get("USDA_API_KEY")
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

# Listas para almacenar los estados con y sin datos
states_with_data = []
states_without_data = []


file_path = f'{source_data_directory}/crop_yield.csv'

def get_crop_yield():
    crop_yield_data = []
    for state_code, state_name in states_dict.items():
        print(f"Llamando API para: {state_name} ({state_code})")
        response = get_usda_quick_stats(api_key=usda_api_key, state_alpha=state_code)
        if response and "data" in response and response["data"]:  # Verifica que "data" no esté vacío
            df = pd.DataFrame(response["data"])  # Convertir en DataFrame
            crop_yield_data.append(df)
            states_with_data.append(state_code)  # Agregar a la lista de estados con datos
        else:
            states_without_data.append(state_code)  # Agregar a la lista de estados sin datos
    print("Datos recopilados exitosamente.")
    crop_yield_df = pd.concat(crop_yield_data, ignore_index=True)
    
    print("\nEstados con datos disponibles:", states_with_data)
    print("\nEstados sin datos disponibles:", states_without_data)
    return crop_yield_df

def save_crop_yield_data(df_crop_yield):
    df_crop_yield.to_csv(file_path)

def read_crop_yied_data():
    df_crop_yield = pd.read_csv(file_path)
    return df_crop_yield
