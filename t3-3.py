import pandas as pd
import numpy as np

# Create synthetic park data
park_data = {
    'park_name': ['Central Park', 'Riverside Park', 'Washington Park', 'Highland Park', 'Lakeview Park'],
    'latitude': [40.7829, 40.7993, 40.7362, 40.6855, 40.7025],
    'longitude': [-73.9654, -73.9779, -73.9903, -73.9956, -74.0121],
    'area_acres': [843, 267, 135, 90, 45] # Park size
}
df_parks = pd.DataFrame(park_data)

# Create tile data (if not already created from previous code)
np.random.seed(42)
tile_ids = [f"Tile_{i}" for i in range(100)]
tile_centers = [(np.random.uniform(40.7, 40.8), np.random.uniform(-74.02, -73.94)) for _ in range(100)]

# Create synthetic population density by neighborhood (using our tile_ids from before)
pop_density = {tile_id: np.random.randint(50, 5000) for tile_id in tile_ids}
df_tiles = pd.DataFrame(list(pop_density.items()), columns=['tile_id', 'population_density'])
df_tiles['latitude'] = [center[0] for center in tile_centers]
df_tiles['longitude'] = [center[1] for center in tile_centers]

# Display the created data
print("✅ Park Data:")
print("=" * 30)
print(df_parks)
print(f"\nTotal parks: {len(df_parks)}")

print("\n✅ Tile Data (Population Density):")
print("=" * 40)
print(df_tiles.head(10))
print(f"\nTotal tiles: {len(df_tiles)}")
print(f"Population density range: {df_tiles['population_density'].min()} - {df_tiles['population_density'].max()}")

# Basic statistics
print("\n📊 Statistics:")
print(f"Average park size: {df_parks['area_acres'].mean():.1f} acres")
print(f"Average population density: {df_tiles['population_density'].mean():.1f} people per tile")