import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
columns = ['state_name', 'county_name', 'state_alpha', 'county_code', 'month', 'year', 'stationTriplet', 'stationId','name', 'lat_centroid', 'lon_centroid', 'TMAX', 'TMIN', 'phh2o', 'ocd', 'cec','sand', 'silt', 'clay', 'PRCP', 'SMS_-2', 'SMS_-4', 'SMS_-8', 'SMS_-20', 'SMS_-40', 'TAVG', 'Value', 'unit_desc']

## Hacemos merge entre los dataframes de clima y suelo.
def merge_monthly_scan_stations_with_soil(soil_data_by_station, monthly_climate_data_by_station):
    monthly_climate_soil_data_by_station = soil_data_by_station.merge(monthly_climate_data_by_station, how = 'inner', on = ['stationTriplet', 'latitude', 'longitude'])
    return monthly_climate_soil_data_by_station

def save_monthly_climate_soil_data_by_station(monthly_climate_soil_data_by_station):
    monthly_climate_soil_data_by_station.to_csv(f'{source_data_directory}/monthly_climate_soil_data_by_station.csv', index=False)

def merge_counties_crop_yield_with_scan_stations(cornbelt_yield_county_centroids, counties_nearest_usda_station):
    crop_yield_usda_stations = cornbelt_yield_county_centroids.merge(right=counties_nearest_usda_station, on=["lat_centroid", "lon_centroid"])
    return crop_yield_usda_stations

def save_crop_yield_scan_stations(crop_yield_scan_stations):
    crop_yield_scan_stations.to_csv(f'{source_data_directory}/crop_yield_scan_stations.csv')
    return crop_yield_scan_stations

def merge_counties_crop_yield_with_historical_scan_stations(crop_yield_usda_stations, monthly_climate_soil_data_by_station):
    monthly_historical_climate_soil_data_by_scan_stations = crop_yield_usda_stations.merge(right=monthly_climate_soil_data_by_station, on=["year", "stationTriplet"])
    monthly_historical_climate_soil_data_by_scan_stations = monthly_historical_climate_soil_data_by_scan_stations[columns]
    monthly_historical_climate_soil_data_by_scan_stations.rename(columns={"Value": "Yield"})
    return monthly_historical_climate_soil_data_by_scan_stations

def save_counties_crop_yield_with_historical_scan_stations(monthly_historical_climate_soil_data_by_station):
    monthly_historical_climate_soil_data_by_station.to_csv(f"{source_data_directory}/monthly_historical_climate_soil_data_by_scan_stations.csv")

