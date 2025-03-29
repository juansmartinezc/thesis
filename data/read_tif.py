import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Ruta de tu archivo descargado (ajústala según tu caso)
archivo_tif = "./out.tif"

# Cargar el raster
with rasterio.open(archivo_tif) as src:
    data = src.read(1)  # Leer la primera banda

# Verificar valores mínimos y máximos
print(f"Min: {np.min(data)}, Max: {np.max(data)}")

# Normalizar la imagen si es necesario (para mejorar la visualización)
data_normalizada = (data - np.min(data)) / (np.max(data) - np.min(data))

# Mostrar la imagen ajustada
plt.figure(figsize=(10, 6))
plt.imshow(data_normalizada, cmap="viridis")  # Puedes probar otros cmap como "terrain" o "plasma"
plt.colorbar(label="Valor del suelo")
plt.title("Datos de SoilGrids")
plt.show()
