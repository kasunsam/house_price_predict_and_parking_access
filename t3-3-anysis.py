import pandas as pd
import numpy as np
import folium

# Create synthetic park data
park_data = {
    'park_name': ['Central Park', 'Riverside Park', 'Washington Park', 'Highland Park', 'Lakeview Park'],
    'latitude': [40.7829, 40.7993, 40.7362, 40.6855, 40.7025],
    'longitude': [-73.9654, -73.9779, -73.9903, -73.9956, -74.0121],
    'area_acres': [843, 267, 135, 90, 45]
}
df_parks = pd.DataFrame(park_data)

# Create tile data
np.random.seed(42)
tile_ids = [f"Tile_{i}" for i in range(100)]
tile_centers = [(np.random.uniform(40.7, 40.8), np.random.uniform(-74.02, -73.94)) for _ in range(100)]

# Create synthetic population density
pop_density = {tile_id: np.random.randint(50, 5000) for tile_id in tile_ids}
df_tiles = pd.DataFrame(list(pop_density.items()), columns=['tile_id', 'population_density'])
df_tiles['latitude'] = [center[0] for center in tile_centers]
df_tiles['longitude'] = [center[1] for center in tile_centers]

# Create a base map, centered on the average location
city_center = [df_parks['latitude'].mean(), df_parks['longitude'].mean()]
m = folium.Map(location=city_center, zoom_start=12, tiles='OpenStreetMap')

# Add parks to the map
for idx, row in df_parks.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['area_acres'] / 50, # Scale the radius by park area
        popup=f"<b>{row['park_name']}</b><br>Area: {row['area_acres']} acres",
        color='green',
        fill=True,
        fillColor='green',
        fillOpacity=0.6
    ).add_to(m)

# Add a choropleth layer for population density
for idx, row in df_tiles.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['population_density'] / 500, # Scale radius by density
        popup=f"Pop Density: {row['population_density']} ppl/sqkm",
        color='blue',
        fill=True,
        fillColor='blue',
        fillOpacity=0.1,
        weight=0
    ).add_to(m)

# Display the map
m.save('park_access_map.html')
print("✅ Interactive map saved as 'park_access_map.html'")
print("📍 Open the file in your web browser to view the map")