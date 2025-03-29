import os
import time
import requests
import pandas as pd

########################################
# CONFIGURACIÓN
########################################

# Lista de propiedades que quieres consultar en SoilGrids
SOIL_PROPERTIES = [
     "silt", "clay", "nitrogen", "sand", "soc"
]

# Nombre de las columnas en tu CSV que guardan lat y lon
LAT_COL = "latitude"
LON_COL = "longitude"

# Archivos de entrada y salida
INPUT_EXCEL = "./stations.xlsx"      # <-- Ajusta según tu archivo real
OUTPUT_CSV = "antenas_scan_soil.csv"

# Número máximo de reintentos cuando ocurre un error 429 o de conexión
MAX_RETRIES = 5

# Tiempo fijo de espera tras cada llamada exitosa (para no saturar la API)
FIXED_SLEEP_BETWEEN_CALLS = 3


########################################
# FUNCIÓN PARA CONSULTAR LA API
########################################

def get_soil_data(lat, lon, properties, max_retries=5):
    """
    Llama a la API de SoilGrids para obtener valores de múltiples propiedades (ej: 'bdod','cec','clay',...)
    en la capa de 0 a 30 cm.
    
    - lat, lon: coordenadas de la antena.
    - properties: lista de strings con los nombres de propiedades a consultar.
    - max_retries: cuántos reintentos hacer si recibimos un 429 o una excepción de conexión.
    
    Devuelve un diccionario, por ejemplo:
    {
      "bdod_0_30cm": 123.4,
      "cec_0_30cm": 10.2,
      "clay_0_30cm": None,
      ...
    }
    """
    # Construir query de múltiples propiedades
    # Se arma como: ?property=bdod&property=cec&property=clay ...
    prop_query = "&".join([f"property={prop}" for prop in properties])
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&{prop_query}"
    print(url)
    delay = 2  # Espera inicial si recibimos 429
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=10)
            print(response)
            if response.status_code == 200:
                data = response.json()
                layers = data.get("properties", {}).get("layers", [])
                
                # Preparar un diccionario para almacenar resultados
                results = {}
                # Inicializar todas las props en None
                for prop in properties:
                    results[f"{prop}_0_30cm"] = None
                
                # Recorrer cada capa devuelta por la API
                # Cada capa es un dict con "name": <prop>, "unit":..., "depths": [...]
                for layer in layers:
                    layer_name = layer.get("name", "")
                    
                    # Solo nos interesan las capas que coincidan con alguna de las props
                    if layer_name in properties:
                        depths = layer.get("depths", [])
                        mean_values = []
                        
                        # Extraer la subcapa para 0-30 cm
                        for d in depths:
                            top = d["range"]["top_depth"]
                            bottom = d["range"]["bottom_depth"]
                            if bottom > 0 and top < 30:
                                val = d["values"].get("mean", None)
                                if val is not None:
                                    mean_values.append(val)
                        
                        if mean_values:
                            avg_val = round(sum(mean_values) / len(mean_values), 2)
                            results[f"{layer_name}_0_30cm"] = avg_val
                
                return results
            
            elif response.status_code == 429:
                # Rate limit
                print(f"Recibido 429 (Too Many Requests) en intento {attempt}/{max_retries} para {lat}, {lon}")
                
                # Chequeamos Retry-After
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        pass
                
                if attempt < max_retries:
                    print(f"Esperando {delay} seg antes de reintentar...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print("Se alcanzó el número máximo de reintentos con 429.")
                    # Devuelve None para todas las propiedades
                    return {f"{prop}_0_30cm": None for prop in properties}
            
            else:
                # Otro código HTTP
                print(f"Respuesta inesperada: {response.status_code} (Intento {attempt}/{max_retries}).")
                # Podríamos imprimir response.text si necesitamos más debug
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    return {f"{prop}_0_30cm": None for prop in properties}
        
        except requests.exceptions.RequestException as e:
            print(f"Excepción en la solicitud (Intento {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return {f"{prop}_0_30cm": None for prop in properties}

    # Si se agotan los reintentos sin éxito
    return {f"{prop}_0_30cm": None for prop in properties}


########################################
# PROGRAMA PRINCIPAL
########################################

def main():
    # 1. Leer CSV de antenas
    df = pd.read_excel(INPUT_EXCEL)
    
    # 2. Agregar las columnas para cada propiedad, si no existen
    for prop in SOIL_PROPERTIES:
        col_name = f"{prop}_0_30cm"
        if col_name not in df.columns:
            df[col_name] = None
    
    # 3. Recorrer cada fila y, si falta alguna de las propiedades, consultarlas todas
    for i, row in df.iterrows():
        # Revisamos si ya existe valor de TODAS las props
        # Si *todas* están definidas, no hacemos la llamada
        all_props_filled = True
        for prop in SOIL_PROPERTIES:
            col_name = f"{prop}_0_30cm"
            if pd.isna(row[col_name]):
                all_props_filled = False
                break
        
        if all_props_filled:
            continue  # no llamamos a la API
        
        # Faltan datos, así que consultamos
        lat = row[LAT_COL]
        lon = row[LON_COL]
        print(row)
        print(lat)
        print(lon)
        # Llamada a la API
        props_dict = get_soil_data(lat, lon, SOIL_PROPERTIES, max_retries=MAX_RETRIES)
        
        # Rellenar el DataFrame con los valores devueltos
        for prop, val in props_dict.items():
            df.at[i, prop] = val
        
        # Pausa para evitar saturar la API
        time.sleep(FIXED_SLEEP_BETWEEN_CALLS)
    
    # 4. Guardar el resultado
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Datos de suelo guardados en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
