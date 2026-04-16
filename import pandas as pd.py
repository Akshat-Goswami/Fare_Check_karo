import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
from google.colab import files
files.upload()

trip_data = pd.read_csv('uber.csv')
print(f"Loaded dataset → {trip_data.shape[0]:,} rows, {trip_data.shape[1]} columns")
print(trip_data.head())

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
trip_data.dropna(inplace=True)
trip_data.drop_duplicates(inplace=True)

# Keep realistic fare values
trip_data = trip_data[(trip_data['fare_amount'] > 0) & (trip_data['fare_amount'] < 500)]

# Sanity check on GPS coordinates
coord_mask = (
    trip_data['pickup_longitude'].between(-180, 180) &
    trip_data['pickup_latitude'].between(-90, 90) &
    trip_data['dropoff_longitude'].between(-180, 180) &
    trip_data['dropoff_latitude'].between(-90, 90)
)
trip_data = trip_data[coord_mask]

# Valid passenger counts only
trip_data = trip_data[trip_data['passenger_count'].between(1, 6)]

print(f"\nAfter cleaning → {trip_data.shape[0]:,} rows remain")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two coordinate pairs."""
    EARTH_RADIUS_KM = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

trip_data['trip_distance_km'] = compute_haversine(
    trip_data['pickup_latitude'],  trip_data['pickup_longitude'],
    trip_data['dropoff_latitude'], trip_data['dropoff_longitude']
)

# Parse datetime features
trip_data['pickup_datetime'] = pd.to_datetime(trip_data['pickup_datetime'])
trip_data['pickup_hour']     = trip_data['pickup_datetime'].dt.hour
trip_data['weekday']         = trip_data['pickup_datetime'].dt.dayofweek
trip_data['pickup_month']    = trip_data['pickup_datetime'].dt.month
trip_data['pickup_year']     = trip_data['pickup_datetime'].dt.year

# Peak-time flags
RUSH_HOURS  = [7, 8, 9, 17, 18, 19]
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))

trip_data['peak_hour_flag']  = trip_data['pickup_hour'].isin(RUSH_HOURS).astype(int)
trip_data['late_night_flag'] = trip_data['pickup_hour'].isin(NIGHT_HOURS).astype(int)

# Remove zero-distance and extreme outlier trips
trip_data = trip_data[trip_data['trip_distance_km'].between(0.1, 200)]

print(f"Feature engineering done → {trip_data.shape[0]:,} rows in final dataset\n")

# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ride Fare Analysis — Exploratory Overview', fontsize=15, fontweight='bold')

# Fare distribution
axes[0, 0].hist(trip_data['fare_amount'], bins=60, color='#4e79a7', edgecolor='white')
axes[0, 0].set_title('Distribution of Fare Amounts')
axes[0, 0].set_xlabel('Fare (USD)')
axes[0, 0].set_ylabel('Frequency')

# Trip distance vs fare
axes[0, 1].scatter(trip_data['trip_distance_km'], trip_data['fare_amount'],
                   alpha=0.25, s=4, color='#59a14f')
axes[0, 1].set_title('Trip Distance vs Fare')
axes[0, 1].set_xlabel('Distance (km)')
axes[0, 1].set_ylabel('Fare (USD)')

# Average fare per hour
avg_fare_by_hour = trip_data.groupby('pickup_hour')['fare_amount'].mean()
axes[1, 0].bar(avg_fare_by_hour.index, avg_fare_by_hour.values,
               color='#f28e2b', edgecolor='white')
axes[1, 0].set_title('Mean Fare by Pickup Hour')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Mean Fare (USD)')

# Fare spread by passenger count
fare_groups = [trip_data[trip_data['passenger_count'] == n]['fare_amount'].values
               for n in range(1, 7)]
axes[1, 1].boxplot(fare_groups, labels=range(1, 7), patch_artist=True,
                   boxprops=dict(facecolor='#edc948', color='gray'))
axes[1, 1].set_title('Fare Distribution by Passenger Count')
axes[1, 1].set_xlabel('Passengers')
axes[1, 1].set_ylabel('Fare (USD)')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved → eda_overview.png\n")

# Correlation heatmap
corr_cols = ['fare_amount', 'trip_distance_km', 'passenger_count',
             'pickup_hour', 'weekday', 'peak_hour_flag', 'late_night_flag']
plt.figure(figsize=(9, 7))
sns.heatmap(trip_data[corr_cols].corr(), annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Correlation matrix saved → correlation_matrix.png\n")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING — RANDOM FOREST
# ─────────────────────────────────────────────
INPUT_FEATURES = [
    'trip_distance_km', 'passenger_count', 'pickup_hour',
    'weekday', 'pickup_month', 'peak_hour_flag', 'late_night_flag'
]

X = trip_data[INPUT_FEATURES]
y = trip_data['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# Scale features
normalizer = MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm  = normalizer.transform(X_test)

# Random Forest with tuned hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=7
)
rf_model.fit(X_train_norm, y_train)

y_pred = rf_model.predict(X_test_norm)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 46)
print("  MODEL EVALUATION — Random Forest Regressor")
print("=" * 46)
print(f"  MAE  : ${mae:.2f}")
print(f"  RMSE : ${rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:2000], y_pred[:2000], alpha=0.3, s=7, color='#4e79a7')
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.8, label='Ideal Fit')
plt.xlabel('Actual Fare (USD)')
plt.ylabel('Predicted Fare (USD)')
plt.title('Predicted vs Actual Fare — Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=INPUT_FEATURES).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#59a14f', edgecolor='white')
plt.title('Feature Importances — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
plt.show()
print("Evaluation plots saved.\n")

# ─────────────────────────────────────────────
# 7. FARE PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_trip_fare(distance_km, num_passengers=1, hour=12,
                      weekday=1, month=6):
    """
    Predict ride fare using the trained Random Forest model.

    Args:
        distance_km    (float) : trip distance in kilometres
        num_passengers (int)   : number of riders (1–6)
        hour           (int)   : pickup hour in 24h format (0–23)
        weekday        (int)   : day of week — 0 = Monday, 6 = Sunday
        month          (int)   : calendar month (1–12)

    Returns:
        float : estimated fare in USD
    """
    is_peak  = 1 if hour in RUSH_HOURS  else 0
    is_night = 1 if hour in NIGHT_HOURS else 0

    input_arr = np.array([[distance_km, num_passengers, hour,
                           weekday, month, is_peak, is_night]])
    input_norm    = normalizer.transform(input_arr)
    estimated_fare = rf_model.predict(input_norm)[0]
    return round(max(estimated_fare, 2.50), 2)

# ─────────────────────────────────────────────
# 8. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("=" * 50)
print("  SAMPLE FARE PREDICTIONS")
print("=" * 50)

sample_trips = [
    {"distance_km": 3.0,  "hour": 8,  "label": "3 km  │ Morning rush"},
    {"distance_km": 10.0, "hour": 14, "label": "10 km │ Midday ride"},
    {"distance_km": 5.0,  "hour": 23, "label": "5 km  │ Late-night trip"},
    {"distance_km": 20.0, "hour": 18, "label": "20 km │ Evening rush"},
]

for trip in sample_trips:
    fare = predict_trip_fare(trip['distance_km'], hour=trip['hour'])
    print(f"  {trip['label']:<28} →  ${fare}")

print()