
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# Load dataset
df = pd.read_csv("flood.csv")

target = "FloodProbability"

# Define the 4 user-friendly groups
climate_features = [
    "MonsoonIntensity",
    "ClimateChange",
    "Deforestation",
    "Landslides"
]

water_features = [
    "TopographyDrainage",
    "RiverManagement",
    "Siltation",
    "Watersheds",
    "WetlandLoss",
    "CoastalVulnerability"
]

infrastructure_features = [
    "DamsQuality",
    "DrainageSystems",
    "DeterioratingInfrastructure",
    "InadequatePlanning",
    "IneffectiveDisasterPreparedness"
]

human_features = [
    "Urbanization",
    "AgriculturalPractices",
    "Encroachments",
    "PopulationScore",
    "PoliticalFactors"
]

# Create group scores using the average of related features
df["Climate_Weather"] = df[climate_features].mean(axis=1)
df["Water_Drainage"] = df[water_features].mean(axis=1)
df["Infrastructure_Preparedness"] = df[infrastructure_features].mean(axis=1)
df["Population_Human_Impact"] = df[human_features].mean(axis=1)

group_features = [
    "Climate_Weather",
    "Water_Drainage",
    "Infrastructure_Preparedness",
    "Population_Human_Impact"
]

# Display groups
print("=" * 70)
print("GROUPED FEATURE TEST")
print("=" * 70)

print("\nClimate & Weather:")
print(climate_features)

print("\nWater & Drainage:")
print(water_features)

print("\nInfrastructure & Preparedness:")
print(infrastructure_features)

print("\nPopulation & Human Impact:")
print(human_features)

# Display group ranges
print("\n" + "=" * 70)
print("GROUP SCORE RANGES")
print("=" * 70)

for feature in group_features:
    print(
        f"{feature}: "
        f"Min = {df[feature].min():.2f}, "
        f"Max = {df[feature].max():.2f}, "
        f"Mean = {df[feature].mean():.2f}"
    )

# Correlation with target


correlation = df[group_features + [target]].corr()[target]
print(correlation.sort_values(ascending=False))

# Prepare data
X = df[group_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print(f"MAE  : {rf_mae:.6f}")
print(f"MSE  : {rf_mse:.6f}")
print(f"RMSE : {rf_rmse:.6f}")
print(f"R²   : {rf_r2:.6f}")

# XGBoost


xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"MAE  : {xgb_mae:.6f}")
print(f"MSE  : {xgb_mse:.6f}")
print(f"RMSE : {xgb_rmse:.6f}")
print(f"R²   : {xgb_r2:.6f}")

# Group importance


importance = pd.DataFrame({
    "Group": group_features,
    "Importance": xgb.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))

# Final comparison

print("Original 20-feature XGBoost R²: 0.916979")
print(f"4-group XGBoost R²:             {xgb_r2:.6f}")
print(f"Difference in R²:               {0.916979 - xgb_r2:.6f}")

# Recommendation

if xgb_r2 >= 0.80:
    print("The 4 grouped inputs retain strong predictive performance.")
elif xgb_r2 >= 0.70:
    print("The 4 grouped inputs provide reasonable predictive performance.")
elif xgb_r2 >= 0.50:
    print("The grouped inputs have moderate performance.")
else:
    print("The grouped inputs lose too much predictive information.")



