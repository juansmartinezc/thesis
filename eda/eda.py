import pandas as pd
from dataprep.eda import create_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = 'historical_monthly_climate_by_scan_stations_sms8_imputed.csv'
source_df = pd.to_csv(file_path)

report = create_report(source_df)


