import pandas as pd
import numpy as np

# Create a synthetic grid of city areas (tiles)
np.random.seed(42)
tile_ids = [f"Tile_{i}" for i in range(100)]
tile_centers = [(np.random.uniform(40.7, 40.8), np.random.uniform(-74.02, -73.94)) for _ in range(100)]

# Create a synthetic list of POIs, each assigned to a tile and a type
poi_types = ['restaurant', 'bar', 'office', 'park', 'school', 'hospital', 'retail']
poi_data = []

for tile_id, center in zip(tile_ids, tile_centers):
    # Each tile has a random number of POIs
    num_pois = np.random.poisson(5) 
    for _ in range(num_pois):
        poi_type = np.random.choice(poi_types, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]) # Probabilities for each type
        # Jitter the location slightly around the tile center
        lat = center[0] + np.random.normal(0, 0.001)
        lon = center[1] + np.random.normal(0, 0.001)
        poi_data.append({'tile_id': tile_id, 'poi_type': poi_type, 'latitude': lat, 'longitude': lon})

df_poi = pd.DataFrame(poi_data)
print("POI Data created successfully!")
print(f"Total POIs: {len(df_poi)}")
print("\nFirst 5 rows:")
print(df_poi.head())
print("\nPOI Type distribution:")
print(df_poi['poi_type'].value_counts())