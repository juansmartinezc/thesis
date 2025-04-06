import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
counties_with_centroid = pd.read_csv(f'{source_data_directory}/counties_centroids_cornbelt.csv')
stations_voronoi = pd.read_csv(f'{source_data_directory}/counties_usda_station.csv')

crop_yield_usda_stations = counties_with_centroid.merge(right=stations_voronoi, on=["lat_centroid", "lon_centroid"])
print(crop_yield_usda_stations.head())
crop_yield_usda_stations.to_csv(f'{source_data_directory}/crop_yield_usda_stations.csv')

monthly_climate_soil_data_by_station = pd.read_csv(f"{source_data_directory}/monthly_climate_soil_data_by_station.csv")
monthly_historical_climate_soil_data_by_station = crop_yield_usda_stations.merge(right=monthly_climate_soil_data_by_station, on=["year", "stationTriplet"])
columns = ['state_name', 'county_name', 'state_alpha', 'county_code', 'month', 'year', 'stationTriplet', 'stationId','name', 'lat_centroid', 'lon_centroid', 'TMAX', 'TMIN', 'phh2o', 'ocd', 'cec','sand', 'silt', 'clay', 'PRCP', 'SMS_-2', 'SMS_-4', 'SMS_-8', 'SMS_-20', 'SMS_-40', 'TAVG', 'Value', 'unit_desc']

monthly_historical_climate_soil_data_by_station = monthly_historical_climate_soil_data_by_station[columns]
monthly_historical_climate_soil_data_by_station.rename(columns={"Value": "Yield"})
monthly_historical_climate_soil_data_by_station.to_csv(f"{source_data_directory}/monthly_historical_climate_soil_data_by_station.csv")




