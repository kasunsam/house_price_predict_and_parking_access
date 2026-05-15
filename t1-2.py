import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset of taxi drop-offs
np.random.seed(42) # For reproducibility

# Cluster 1: Downtown area
n_downtown = 150
downtown_lat = np.random.normal(40.75, 0.01, n_downtown)
downtown_lon = np.random.normal(-73.99, 0.01, n_downtown)

# Cluster 2: Midtown area
n_midtown = 100
midtown_lat = np.random.normal(40.76, 0.008, n_midtown)
midtown_lon = np.random.normal(-73.98, 0.008, n_midtown)

# Noise: Random drop-offs across the city
n_noise = 50
noise_lat = np.random.uniform(40.70, 40.80, n_noise)
noise_lon = np.random.uniform(-74.02, -73.94, n_noise)

# Combine all points
latitudes = np.concatenate([downtown_lat, midtown_lat, noise_lat])
longitudes = np.concatenate([downtown_lon, midtown_lon, noise_lon])

# Create a DataFrame
df_taxi = pd.DataFrame({'latitude': latitudes, 'longitude': longitudes})
print(df_taxi.head())