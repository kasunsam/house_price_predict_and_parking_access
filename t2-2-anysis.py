import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Display information about the created dataset
print("✅ POI Data created successfully!")
print("=" * 50)
print(f"Total POIs: {len(df_poi)}")
print(f"Total tiles: {len(tile_ids)}")
print(f"POI Types: {poi_types}")

print("\nFirst 10 rows:")
print(df_poi.head(10))

print("\n📊 POI Type Distribution:")
type_counts = df_poi['poi_type'].value_counts()
print(type_counts)

print("\n📈 POIs per Tile Statistics:")
pois_per_tile = df_poi['tile_id'].value_counts()
print(f"Average POIs per tile: {pois_per_tile.mean():.2f}")
print(f"Min POIs per tile: {pois_per_tile.min()}")
print(f"Max POIs per tile: {pois_per_tile.max()}")

# Visualize the data
plt.figure(figsize=(15, 5))

# Plot 1: POI locations
plt.subplot(1, 3, 1)
colors = {'restaurant': 'red', 'bar': 'blue', 'office': 'green', 'park': 'lightgreen', 
          'school': 'orange', 'hospital': 'pink', 'retail': 'purple'}

for poi_type in poi_types:
    mask = df_poi['poi_type'] == poi_type
    plt.scatter(df_poi[mask]['longitude'], df_poi[mask]['latitude'], 
               c=colors[poi_type], label=poi_type, alpha=0.6, s=30)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('POI Locations by Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot 2: POI type distribution
plt.subplot(1, 3, 2)
type_counts.plot(kind='bar', color=[colors[t] for t in type_counts.index])
plt.title('POI Type Distribution')
plt.xlabel('POI Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 3: POIs per tile distribution
plt.subplot(1, 3, 3)
plt.hist(pois_per_tile.values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of POIs per Tile')
plt.xlabel('Number of POIs')
plt.ylabel('Number of Tiles')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save to CSV
df_poi.to_csv('synthetic_poi_data.csv', index=False)
print(f"\n💾 Data saved to 'synthetic_poi_data.csv'")