
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Load dataset
df = pd.read_csv("flood.csv")

# Target
target = "FloodProbability"


# Define the 4 groups
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


# Create the 4 group scores
df["Climate_Weather"] = df[climate_features].mean(axis=1)

df["Water_Drainage"] = df[water_features].mean(axis=1)

df["Infrastructure_Preparedness"] = (
    df[infrastructure_features].mean(axis=1)
)

df["Population_Human_Impact"] = (
    df[human_features].mean(axis=1)
)


# Final model inputs
group_features = [
    "Climate_Weather",
    "Water_Drainage",
    "Infrastructure_Preparedness",
    "Population_Human_Impact"
]

X = df[group_features]
y = df[target]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train final XGBoost model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

print("Training final XGBoost model...")

model.fit(X_train, y_train)


# Evaluate model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# Display the model inputs
print("\nFinal model inputs:")

for feature in group_features:
    print(f"- {feature}")


# Save the trained model
joblib.dump(model, "final_model.pkl")

print("\nFinal model saved as:")
print("final_model.pkl")

print("\nModel training completed successfully.")

