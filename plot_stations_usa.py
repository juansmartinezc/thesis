import pandas as pd
import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point, Polygon
from geovoronoi import voronoi_regions_from_coords, points_to_coords

# 1. Cargar datos de las estaciones
df = pd.read_csv("stations.csv")

# 2. Convertir a GeoDataFrame con coordenadas WGS84
points = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf_points = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

# 3. Cargar el contorno de EE.UU. (shapefile o GeoJSON)
#    - Asegúrate de que la ruta y el nombre del archivo son correctos.
gdf_usa = gpd.read_file("shape_files/state/cb_2018_us_state_20m.shp")  # o "usa.geojson"

# 4. Convertir a la misma proyección (WGS84) si no lo está ya
#    (solo si tu shapefile no está en EPSG:4326)
gdf_usa = gdf_usa.to_crs("EPSG:4326")

# 5. Unir todas las geometrías en una sola (si son varios polígonos)
#    - Esto devuelve un shapely.MultiPolygon o Polygon, según el caso.
usa_polygon = gdf_usa.unary_union

# 6. Calcular polígonos Voronoi sobre el polígono de EE.UU.
region_polys, region_pts = voronoi_regions_from_coords(
    points_to_coords(points),
    usa_polygon  # <-- aquí usamos el polígono real de EE.UU.
)

# 7. Crear un GeoDataFrame con los polígonos de Voronoi resultantes
voronoi_list = []
for i, poly in region_polys.items():
    station_id = df.loc[i, 'stationTriplet']
    voronoi_list.append({
        'stationTriplet': station_id,
        'geometry': poly
    })

gdf_voronoi = gpd.GeoDataFrame(voronoi_list, crs="EPSG:4326")

# 8. (Opcional) Asignar un color único (columna 'dummy') para pintarlos en blanco
gdf_voronoi['dummy'] = 'Voronoi'

# 9. Convertir a GeoJSON para plotear en Plotly
geojson_voronoi = gdf_voronoi.__geo_interface__

# 10. Graficar
fig = px.choropleth_mapbox(
    gdf_voronoi,
    geojson=geojson_voronoi,
    locations=gdf_voronoi.index,
    color='dummy',  # usamos la columna 'dummy'
    color_discrete_map={'Voronoi':'white'},  # todos blancos
    mapbox_style="carto-positron",
    # Ajustar el centro y zoom si lo deseas (un centrado aproximado)
    center={"lat": 39.5, "lon": -98.35},
    zoom=3,
    opacity=0.3
)

# Ajustar contorno de los polígonos
fig.update_traces(
    marker_line_width=1,
    marker_line_color='black',
    showscale=False
)

# Añadir las estaciones
fig.add_scattermapbox(
    lat=df['latitude'],
    lon=df['longitude'],
    mode='markers',
    marker=dict(size=6, color='red'),
    text=df['stationTriplet'],
    hoverinfo='text',
    name='Estaciones'
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
