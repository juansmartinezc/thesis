import requests
import json

# Coordenadas del punto (latitud y longitud)
lat = 4.5709   # por ejemplo, Colombia
lon = -74.2973

# URL base de la API
url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Parámetros
params = {
    "lat": lat,
    "lon": lon,
    "property": ["phh2o", "ocd", "cec", "sand", "silt", "clay"],  # propiedades del suelo
    "depth": ["0-5cm", "5-15cm"],  # profundidades
}

# Realizar la petición
response = requests.get(url, params=params)

# Verificar si la respuesta fue exitosa
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))  # imprimir bonito
else:
    print("Error:", response.status_code, response.text)
