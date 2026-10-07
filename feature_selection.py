
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("FLOOD PREDICTION - FEATURE SELECTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv("flood.csv")

# Target variable
target = "FloodProbability"

# Separate input features and target
X = df.drop(columns=[target])
y = df[target]

print(f"Dataset shape: {df.shape}")
print(f"Number of input features: {X.shape[1]}")


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 3. RANDOM FOREST - ALL 20 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - ALL 20 FEATURES")
print("=" * 70)

rf_all = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_all.fit(X_train, y_train)

rf_all_pred = rf_all.predict(X_test)

rf_all_mae = mean_absolute_error(y_test, rf_all_pred)
rf_all_mse = mean_squared_error(y_test, rf_all_pred)
rf_all_rmse = np.sqrt(rf_all_mse)
rf_all_r2 = r2_score(y_test, rf_all_pred)

print(f"MAE  : {rf_all_mae:.6f}")
print(f"MSE  : {rf_all_mse:.6f}")
print(f"RMSE : {rf_all_rmse:.6f}")
print(f"R²   : {rf_all_r2:.6f}")


# ============================================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_all.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(rf_importance.to_string(index=False))


# ============================================================
# 5. XGBOOST - ALL 20 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST - ALL 20 FEATURES")
print("=" * 70)

xgb_all = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_all.fit(X_train, y_train)

xgb_all_pred = xgb_all.predict(X_test)

xgb_all_mae = mean_absolute_error(y_test, xgb_all_pred)
xgb_all_mse = mean_squared_error(y_test, xgb_all_pred)
xgb_all_rmse = np.sqrt(xgb_all_mse)
xgb_all_r2 = r2_score(y_test, xgb_all_pred)

print(f"MAE  : {xgb_all_mae:.6f}")
print(f"MSE  : {xgb_all_mse:.6f}")
print(f"RMSE : {xgb_all_rmse:.6f}")
print(f"R²   : {xgb_all_r2:.6f}")


# ============================================================
# 6. XGBOOST FEATURE IMPORTANCE
# ============================================================

xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_all.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nXGBoost Feature Importance:")
print(xgb_importance.to_string(index=False))


# ============================================================
# 7. SELECT TOP 5 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 FEATURES")
print("=" * 70)

# Get top 5 from Random Forest
rf_top5 = rf_importance.head(5)["Feature"].tolist()

# Get top 5 from XGBoost
xgb_top5 = xgb_importance.head(5)["Feature"].tolist()

print("\nTop 5 from Random Forest:")
for i, feature in enumerate(rf_top5, 1):
    print(f"{i}. {feature}")

print("\nTop 5 from XGBoost:")
for i, feature in enumerate(xgb_top5, 1):
    print(f"{i}. {feature}")


# ============================================================
# 8. FIND COMMON TOP FEATURES
# ============================================================

common_features = [
    feature for feature in rf_top5
    if feature in xgb_top5
]

print("\nCommon features appearing in BOTH Top 5 lists:")
for i, feature in enumerate(common_features, 1):
    print(f"{i}. {feature}")

print(f"\nNumber of common features: {len(common_features)}")


# ============================================================
# 9. CREATE FINAL TOP 5
# ============================================================

# First take common features.
# If fewer than 5 are common, fill the remaining positions
# using the XGBoost ranking.

final_features = common_features.copy()

for feature in xgb_top5:
    if feature not in final_features:
        final_features.append(feature)

    if len(final_features) == 5:
        break

print("\n" + "=" * 70)
print("FINAL 5 FEATURES SELECTED FOR TESTING")
print("=" * 70)

for i, feature in enumerate(final_features, 1):
    print(f"{i}. {feature}")


# ============================================================
# 10. RANDOM FOREST - TOP 5 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - TOP 5 FEATURES")
print("=" * 70)

X_train_5 = X_train[final_features]
X_test_5 = X_test[final_features]

rf_5 = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_5.fit(X_train_5, y_train)

rf_5_pred = rf_5.predict(X_test_5)

rf_5_mae = mean_absolute_error(y_test, rf_5_pred)
rf_5_mse = mean_squared_error(y_test, rf_5_pred)
rf_5_rmse = np.sqrt(rf_5_mse)
rf_5_r2 = r2_score(y_test, rf_5_pred)

print(f"MAE  : {rf_5_mae:.6f}")
print(f"MSE  : {rf_5_mse:.6f}")
print(f"RMSE : {rf_5_rmse:.6f}")
print(f"R²   : {rf_5_r2:.6f}")


# ============================================================
# 11. XGBOOST - TOP 5 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST - TOP 5 FEATURES")
print("=" * 70)

xgb_5 = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_5.fit(X_train_5, y_train)

xgb_5_pred = xgb_5.predict(X_test_5)

xgb_5_mae = mean_absolute_error(y_test, xgb_5_pred)
xgb_5_mse = mean_squared_error(y_test, xgb_5_pred)
xgb_5_rmse = np.sqrt(xgb_5_mse)
xgb_5_r2 = r2_score(y_test, xgb_5_pred)

print(f"MAE  : {xgb_5_mae:.6f}")
print(f"MSE  : {xgb_5_mse:.6f}")
print(f"RMSE : {xgb_5_rmse:.6f}")
print(f"R²   : {xgb_5_r2:.6f}")


# ============================================================
# 12. PERFORMANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Random Forest - 20 Features",
        "Random Forest - Top 5 Features",
        "XGBoost - 20 Features",
        "XGBoost - Top 5 Features"
    ],
    "MAE": [
        rf_all_mae,
        rf_5_mae,
        xgb_all_mae,
        xgb_5_mae
    ],
    "RMSE": [
        rf_all_rmse,
        rf_5_rmse,
        xgb_all_rmse,
        xgb_5_rmse
    ],
    "R2": [
        rf_all_r2,
        rf_5_r2,
        xgb_all_r2,
        xgb_5_r2
    ]
})

print(comparison.to_string(index=False))


# ============================================================
# 13. PERFORMANCE DROP
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE CHANGE")
print("=" * 70)

print(
    f"\nRandom Forest R² change: "
    f"{rf_all_r2:.6f} → {rf_5_r2:.6f}"
)

print(
    f"XGBoost R² change: "
    f"{xgb_all_r2:.6f} → {xgb_5_r2:.6f}"
)


# ============================================================
# 14. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if xgb_5_r2 >= 0.80:
    print("\n✓ The Top 5 features provide strong performance.")
    print("✓ We can consider using these 5 features for the user interface.")
elif xgb_5_r2 >= 0.70:
    print("\n✓ The Top 5 features provide reasonably good performance.")
    print("✓ We can consider using them for a simple user interface.")
else:
    print("\n⚠ The Top 5 features cause a significant performance reduction.")
    print("⚠ We should test additional features before finalizing the user interface.")

print("\nFinal candidate features:")
for i, feature in enumerate(final_features, 1):
    print(f"{i}. {feature}")

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETED")
print("=" * 70)

