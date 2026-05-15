import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Convert price to integer for cleaner display
df_houses['price'] = df_houses['price'].astype(int)

print("🏠 HOUSING PRICE PREDICTION WITH RANDOM FOREST")
print("=" * 50)

# Display basic info about the data
print(f"Dataset shape: {df_houses.shape}")
print(f"Price range: ${df_houses['price'].min():,} - ${df_houses['price'].max():,}")
print("\nFirst 5 rows:")
print(df_houses.head())

# Prepare features and target
X = df_houses[['latitude', 'longitude', 'sq_ft', 'bedrooms']]
y = df_houses['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 DATA SPLIT:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 MODEL PERFORMANCE:")
print(f'Mean Absolute Error: ${mae:,.2f}')
print(f'R² Score: {r2:.4f}')

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

print("\n🔍 FEATURE IMPORTANCES:")
print(feat_imp_df)

# Visualizations
plt.figure(figsize=(15, 5))

# Plot 1: Feature Importance
plt.subplot(1, 3, 1)
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()

# Plot 2: Actual vs Predicted Prices
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted Prices\nR² = {r2:.4f}')

# Plot 3: Prediction Error Distribution
plt.subplot(1, 3, 3)
errors = y_test - y_pred
plt.hist(errors, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title(f'Prediction Error Distribution\nMAE = ${mae:,.2f}')

plt.tight_layout()
plt.show()

# Additional Analysis
print("\n📊 ADDITIONAL ANALYSIS:")
print("=" * 30)

# Predict on training data to check for overfitting
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training MAE: ${train_mae:,.2f}")
print(f"Training R²: {train_r2:.4f}")

if train_r2 - r2 > 0.1:
    print("⚠️  Warning: Model might be overfitting (large gap between train and test R²)")
else:
    print("✅ Model generalization looks good")

# Sample predictions
print(f"\n🎯 SAMPLE PREDICTIONS (First 10 test samples):")
sample_results = pd.DataFrame({
    'Actual_Price': y_test.values[:10],
    'Predicted_Price': y_pred[:10].astype(int),
    'Error': (y_test.values[:10] - y_pred[:10].astype(int))
})
sample_results['Error_Percentage'] = (sample_results['Error'] / sample_results['Actual_Price'] * 100).round(2)
print(sample_results)

# Model insights
print(f"\n💡 MODEL INSIGHTS:")
print(f"• The model explains {r2*100:.1f}% of the variance in housing prices")
print(f"• Average prediction error: ${mae:,.2f}")
print(f"• Most important feature: {feat_imp_df.iloc[0]['feature']} ({feat_imp_df.iloc[0]['importance']*100:.1f}%)")
print(f"• Least important feature: {feat_imp_df.iloc[-1]['feature']} ({feat_imp_df.iloc[-1]['importance']*100:.1f}%)")

# Save predictions
results_df = pd.DataFrame({
    'Actual_Price': y_test.values,
    'Predicted_Price': y_pred.astype(int),
    'Error': (y_test.values - y_pred.astype(int))
})
results_df.to_csv('housing_price_predictions.csv', index=False)
print(f"\n💾 Predictions saved to 'housing_price_predictions.csv'")