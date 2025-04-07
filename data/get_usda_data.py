import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_usda_quick_stats(api_key, source_desc="SURVEY", sector_desc="CROPS", group_desc="FIELD CROPS",
                         commodity_desc="CORN", statisticcat_desc="YIELD", year_ge=2000,
                         agg_level_desc="COUNTY", unit_desc="BU / ACRE", state_alpha="IA"):
    """
    Llama a la API de Quick Stats del USDA para obtener datos agrícolas.

    Parámetros:
        api_key (str): Clave de API de Quick Stats.
        source_desc (str, opcional): Fuente de los datos (por defecto "SURVEY").
        sector_desc (str, opcional): Sector de los datos (por defecto "CROPS").
        group_desc (str, opcional): Grupo de cultivos (por defecto "FIELD CROPS").
        commodity_desc (str, opcional): Producto agrícola (por defecto "CORN").
        statisticcat_desc (str, opcional): Categoría estadística (por defecto "YIELD").
        year_ge (int, opcional): Año mínimo de los datos (por defecto 2020).
        freq_desc (str, opcional): Frecuencia de los datos (por defecto "ANNUAL").
        agg_level_desc (str, opcional): Nivel de agregación de los datos (por defecto "COUNTY").
        unit_desc (str, opcional): Unidad de medida (por defecto "BU / ACRE").
        state_alpha (str, opcional): Estado de EE.UU. en formato de dos letras (por defecto "IA").

    Retorna:
        dict: Respuesta en formato JSON con los datos agrícolas.
    """
    url = os.environ.get("USDA_QUICK_STATS_URL")
    params = {
        "key": api_key,
        "source_desc": source_desc,
        "sector_desc": sector_desc,
        "group_desc": group_desc,
        "commodity_desc": commodity_desc,
        "statisticcat_desc": statisticcat_desc,
        "year__GE": year_ge,
        "agg_level_desc": agg_level_desc,
        "unit_desc": unit_desc,
        "state_alpha": state_alpha
    }

    try:
        response = requests.get(url, params=params)
        print()
        response.raise_for_status()  # Manejo de errores de solicitud
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API: {e}")
        return None
    

