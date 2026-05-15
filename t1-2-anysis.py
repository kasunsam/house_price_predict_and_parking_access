import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. LOAD OR CREATE SAMPLE DATA
# =============================================================================

try:
    # Try to load your taxi data file - UPDATE THIS FILE PATH
    df_taxi = pd.read_csv('taxi_data.csv')  # Change to your actual file name
    print("✅ Data loaded successfully from file")
    
except FileNotFoundError:
    print("📝 No data file found. Creating sample taxi data for demonstration...")
    
    # Create realistic sample taxi data in New York area
    np.random.seed(42)
    n_points = 1000
    
    # Generate realistic coordinates around Manhattan
    manhattan_center = (40.7589, -73.9851)
    
    df_taxi = pd.DataFrame({
        'latitude': np.random.normal(manhattan_center[0], 0.02, n_points),
        'longitude': np.random.normal(manhattan_center[1], 0.02, n_points),
        'passenger_count': np.random.randint(1, 6, n_points),
        'fare_amount': np.random.uniform(5, 50, n_points),
        'trip_duration': np.random.uniform(5, 60, n_points),  # minutes
        'hour_of_day': np.random.randint(0, 24, n_points)
    })
    
    # Add some clear clusters
    cluster_centers = [
        (40.7614, -73.9776),  # Times Square
        (40.7505, -73.9934),  # Penn Station
        (40.6892, -74.0445),  # Statue of Liberty area
        (40.7282, -73.7949),  # JFK Airport
    ]
    
    for i, (lat, lon) in enumerate(cluster_centers):
        cluster_size = 50
        cluster_data = pd.DataFrame({
            'latitude': np.random.normal(lat, 0.005, cluster_size),
            'longitude': np.random.normal(lon, 0.005, cluster_size),
            'passenger_count': np.random.randint(1, 6, cluster_size),
            'fare_amount': np.random.uniform(5, 50, cluster_size),
            'trip_duration': np.random.uniform(5, 60, cluster_size),
            'hour_of_day': np.random.randint(0, 24, cluster_size)
        })
        df_taxi = pd.concat([df_taxi, cluster_data], ignore_index=True)

# =============================================================================
# 2. EXPLORE THE DATA
# =============================================================================

print("\n📊 DATA EXPLORATION:")
print("=" * 50)
print(f"Dataset shape: {df_taxi.shape}")
print(f"Columns: {df_taxi.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_taxi.head())
print("\nBasic statistics:")
print(df_taxi.describe())
print("\nMissing values:")
print(df_taxi.isnull().sum())

# =============================================================================
# 3. PREPARE DATA FOR CLUSTERING
# =============================================================================

print("\n🔄 PREPARING DATA FOR CLUSTERING...")

# Extract coordinates for clustering
coords = df_taxi[['latitude', 'longitude']].values
print(f"Coordinates shape: {coords.shape}")

# Standardize the coordinates (important for DBSCAN)
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
print("Coordinates standardized for DBSCAN")

# =============================================================================
# 4. APPLY DBSCAN CLUSTERING
# =============================================================================

print("\n🎯 APPLYING DBSCAN CLUSTERING...")

# DBSCAN parameters - you can adjust these
eps = 0.3      # Maximum distance between points in the same cluster
min_samples = 5  # Minimum number of points to form a cluster

# Create and fit DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(coords_scaled)

# Add cluster labels to dataframe
df_taxi['cluster'] = clusters

# Analyze clustering results
unique_clusters = np.unique(clusters)
n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
n_noise = np.sum(clusters == -1)

print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Cluster labels: {unique_clusters}")

# =============================================================================
# 5. VISUALIZE RESULTS
# =============================================================================

print("\n📈 VISUALIZING RESULTS...")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Taxi Data Clustering Analysis with DBSCAN', fontsize=16, fontweight='bold')

# Plot 1: Original data
axes[0, 0].scatter(df_taxi['longitude'], df_taxi['latitude'], 
                  c='blue', alpha=0.6, s=30)
axes[0, 0].set_title('Original Taxi Data')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Clustering results
scatter = axes[0, 1].scatter(df_taxi['longitude'], df_taxi['latitude'], 
                            c=df_taxi['cluster'], cmap='tab10', alpha=0.7, s=30)
axes[0, 1].set_title(f'DBSCAN Clustering (Clusters: {n_clusters}, Noise: {n_noise})')
axes[0, 1].set_xlabel('Longitude')
axes[0, 1].set_ylabel('Latitude')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Cluster ID')

# Plot 3: Cluster sizes
if n_clusters > 0:
    cluster_sizes = df_taxi[df_taxi['cluster'] != -1]['cluster'].value_counts().sort_index()
    axes[1, 0].bar(cluster_sizes.index, cluster_sizes.values, color='lightcoral')
    axes[1, 0].set_title('Cluster Sizes')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Number of Points')
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No clusters found', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Cluster Sizes')

# Plot 4: Noise vs Clustered points
noise_vs_clustered = [n_noise, len(df_taxi) - n_noise]
labels = ['Noise Points', 'Clustered Points']
colors = ['red', 'green']
axes[1, 1].pie(noise_vs_clustered, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Noise vs Clustered Points Distribution')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. CLUSTER ANALYSIS
# =============================================================================

print("\n🔍 CLUSTER ANALYSIS:")
print("=" * 50)

if n_clusters > 0:
    # Analyze each cluster
    for cluster_id in range(n_clusters):
        cluster_data = df_taxi[df_taxi['cluster'] == cluster_id]
        print(f"\n📁 Cluster {cluster_id}:")
        print(f"   Size: {len(cluster_data)} points")
        print(f"   Center: ({cluster_data['latitude'].mean():.4f}, {cluster_data['longitude'].mean():.4f})")
        print(f"   Avg passengers: {cluster_data['passenger_count'].mean():.1f}")
        print(f"   Avg fare: ${cluster_data['fare_amount'].mean():.2f}")
        print(f"   Avg trip duration: {cluster_data['trip_duration'].mean():.1f} min")

# Noise points analysis
if n_noise > 0:
    noise_data = df_taxi[df_taxi['cluster'] == -1]
    print(f"\n🚫 Noise Points (Cluster -1):")
    print(f"   Count: {len(noise_data)}")
    print(f"   These are isolated pickup/dropoff locations")

# =============================================================================
# 7. ADDITIONAL VISUALIZATION - CLUSTER CHARACTERISTICS
# =============================================================================

# Create another figure for cluster characteristics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Average passengers per cluster
if n_clusters > 0:
    passenger_means = []
    for cluster_id in range(n_clusters):
        cluster_mean = df_taxi[df_taxi['cluster'] == cluster_id]['passenger_count'].mean()
        passenger_means.append(cluster_mean)
    
    axes[0].bar(range(n_clusters), passenger_means, color='skyblue')
    axes[0].set_title('Average Passengers per Cluster')
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Average Passengers')
else:
    axes[0].text(0.5, 0.5, 'No clusters found', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Average Passengers per Cluster')

# Fare amount distribution by cluster
if n_clusters > 0:
    fare_data = [df_taxi[df_taxi['cluster'] == i]['fare_amount'] for i in range(n_clusters)]
    axes[1].boxplot(fare_data)
    axes[1].set_title('Fare Amount Distribution by Cluster')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Fare Amount ($)')
else:
    axes[1].text(0.5, 0.5, 'No clusters found', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Fare Amount Distribution by Cluster')

# Trip duration by hour of day
scatter = axes[2].scatter(df_taxi['hour_of_day'], df_taxi['trip_duration'], 
                         c=df_taxi['cluster'], cmap='tab10', alpha=0.6)
axes[2].set_title('Trip Duration vs Hour of Day (Colored by Cluster)')
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Trip Duration (min)')
plt.colorbar(scatter, ax=axes[2], label='Cluster ID')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

# Save the clustered data to a new CSV file
output_filename = 'taxi_data_clustered.csv'
df_taxi.to_csv(output_filename, index=False)
print(f"\n💾 Clustered data saved to: {output_filename}")

# =============================================================================
# 9. SUMMARY REPORT
# =============================================================================

print("\n" + "="*60)
print("📋 CLUSTERING ANALYSIS SUMMARY REPORT")
print("="*60)
print(f"Total data points: {len(df_taxi)}")
print(f"Number of clusters identified: {n_clusters}")
print(f"Noise points (outliers): {n_noise} ({n_noise/len(df_taxi)*100:.1f}%)")
print(f"Clustered points: {len(df_taxi) - n_noise} ({(len(df_taxi) - n_noise)/len(df_taxi)*100:.1f}%)")
print(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")

if n_clusters > 0:
    print(f"\nCluster size distribution:")
    for cluster_id in range(n_clusters):
        size = len(df_taxi[df_taxi['cluster'] == cluster_id])
        print(f"  Cluster {cluster_id}: {size} points ({size/len(df_taxi)*100:.1f}%)")

print("\n✅ Analysis complete!")