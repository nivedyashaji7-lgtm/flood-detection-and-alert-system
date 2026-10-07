import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("flood.csv")

print("=" * 70)
print("FLOOD DATASET - DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 2. FIRST 5 ROWS
# ============================================================

print("\n1. FIRST 5 ROWS")
print("-" * 70)
print(df.head())


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\n2. DATASET SHAPE")
print("-" * 70)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n3. COLUMN NAMES")
print("-" * 70)
print(df.columns.tolist())


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n4. DATA TYPES")
print("-" * 70)
print(df.dtypes)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n5. MISSING VALUES")
print("-" * 70)
print(df.isnull().sum())


# ============================================================
# 7. TOTAL MISSING VALUES
# ============================================================

print("\n6. TOTAL MISSING VALUES")
print("-" * 70)
print("Total missing values:", df.isnull().sum().sum())


# ============================================================
# 8. DUPLICATE ROWS
# ============================================================

print("\n7. DUPLICATE ROWS")
print("-" * 70)
print("Number of duplicate rows:", df.duplicated().sum())


# ============================================================
# 9. BASIC STATISTICS
# ============================================================

print("\n8. BASIC STATISTICS")
print("-" * 70)
print(df.describe())


# ============================================================
# 10. MINIMUM VALUES
# ============================================================

print("\n9. MINIMUM VALUES")
print("-" * 70)
print(df.min())


# ============================================================
# 11. MAXIMUM VALUES
# ============================================================

print("\n10. MAXIMUM VALUES")
print("-" * 70)
print(df.max())


# ============================================================
# 12. UNIQUE VALUES
# ============================================================

print("\n11. NUMBER OF UNIQUE VALUES")
print("-" * 70)
print(df.nunique())


# ============================================================
# 13. FLOOD PROBABILITY DISTRIBUTION
# ============================================================

print("\n12. FLOOD PROBABILITY SUMMARY")
print("-" * 70)
print("Minimum Flood Probability:", df["FloodProbability"].min())
print("Maximum Flood Probability:", df["FloodProbability"].max())
print("Average Flood Probability:", df["FloodProbability"].mean())
print("Median Flood Probability:", df["FloodProbability"].median())


# ============================================================
# 14. FLOOD RISK CATEGORIES
# ============================================================

print("\n13. FLOOD RISK CATEGORY DISTRIBUTION")
print("-" * 70)

df["FloodRisk"] = pd.cut(
    df["FloodProbability"],
    bins=[0, 0.4, 0.6, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

print(df["FloodRisk"].value_counts())
print("\nPercentage:")
print(df["FloodRisk"].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 15. CORRELATION WITH FLOOD PROBABILITY
# ============================================================

print("\n14. CORRELATION WITH FLOOD PROBABILITY")
print("-" * 70)

correlation = df.corr(numeric_only=True)["FloodProbability"]
correlation = correlation.sort_values(ascending=False)

print(correlation)


# ============================================================
# 16. TOP 10 CORRELATED FEATURES
# ============================================================

print("\n15. TOP 10 FEATURES CORRELATED WITH FLOOD PROBABILITY")
print("-" * 70)

top_correlation = correlation.drop("FloodProbability").abs().sort_values(ascending=False)

print(top_correlation.head(10))


# ============================================================
# 17. CHECK DATA RANGE OF INPUT FEATURES
# ============================================================

print("\n16. INPUT FEATURE RANGES")
print("-" * 70)

feature_columns = df.drop(columns=["FloodProbability", "FloodRisk"]).columns

for column in feature_columns:
    print(
        f"{column}: "
        f"Min = {df[column].min()}, "
        f"Max = {df[column].max()}, "
        f"Mean = {df[column].mean():.2f}"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("DATA ANALYSIS COMPLETED")
print("=" * 70)