import pandas as pd
import numpy as np

# Create synthetic housing data
np.random.seed(42)
n_houses = 500

# Simulate a city where northern areas are more expensive
base_lat = 40.75
base_lon = -73.98

# House locations
latitudes = np.random.normal(base_lat, 0.03, n_houses)
longitudes = np.random.normal(base_lon, 0.04, n_houses)

# Property features
sq_ft = np.random.normal(2000, 500, n_houses).astype(int)
bedrooms = np.random.choice([2, 3, 4, 5], n_houses, p=[0.1, 0.5, 0.3, 0.1])

# Synthesize Price: Base + size component + bedroom component + location component (based on lat) + noise
base_price = 200_000
price = (base_price +
         (sq_ft - 2000) * 150 + # $150 per sq ft over 2000
         (bedrooms - 3) * 50_000 + # $50k per bedroom over 3
         (latitudes - base_lat) * 5_000_000 + # North is more expensive
         np.random.normal(0, 25_000, n_houses) # Noise
        )

df_houses = pd.DataFrame({
    'latitude': latitudes,
    'longitude': longitudes,
    'sq_ft': sq_ft,
    'bedrooms': bedrooms,
    'price': price
})

print("✅ Synthetic housing data created successfully!")
print("=" * 50)
print(df_houses.head())
print(f"\nTotal houses: {len(df_houses)}")