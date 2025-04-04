import pandas as pd

counties_with_centroid = pd.read_csv('countries_cleaned.csv')
stations_voronoi = pd.read_csv('condados_estacion_voronoi.csv')

crop_yield_centroid_df = counties_with_centroid.merge(right=stations_voronoi, on=["lat_centroid", "lon_centroid"])
print(crop_yield_centroid_df.head())
crop_yield_centroid_df.to_csv("crop_yield_centroid_df.csv")

stations_climate_soil_hist_df = pd.read_csv("result.csv")
final_df = crop_yield_centroid_df.merge(right=stations_climate_soil_hist_df, on=["year", "stationTriplet"])
columns = ['state_name', 'county_name', 'state_alpha', 'county_code', 'month', 'year', 'stationTriplet', 'stationId','name', 'lat_centroid', 'lon_centroid', 'TMAX', 'TMIN', 'phh2o_0_30cm', 'ocd_0_30cm', 'cec_0_30cm','sand_0_30cm', 'silt_0_30cm', 'clay_0_30cm', 'PRCP', 'SMS_-2', 'SMS_-4', 'SMS_-8', 'SMS_-20', 'SMS_-40', 'TAVG', 'Value', 'unit_desc']

final_df = final_df[columns]
final_df.rename(columns={"Value": "Yield"})
final_df.to_csv("final_df.csv")




