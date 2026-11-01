
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

print("\n========== XGBOOST MODEL TRAINING ==========\n")

df = pd.read_csv("f1_driver_features.csv")

df["race_date"] = pd.to_datetime(df["race_date"])
df["position"] = pd.to_numeric(df["position"], errors="coerce")

df = df.dropna(subset=["position", "race_date", "race_id"])
df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

categorical_features = [
    "driver_id",
    "constructor_id",
    "circuit_id"
]

numerical_features = [
    "grid",
    "grid_is_zero",
    "driver_avg_previous",
    "driver_avg_last_5",
    "driver_previous_races",
    "constructor_avg_previous",
    "constructor_avg_last_5",
    "driver_circuit_avg_previous",
    "driver_circuit_previous_races",
    "circuit_length_km"
]

feature_columns = categorical_features + numerical_features

X = df[feature_columns].copy()
y = df["position"].copy()


# ==========================================
# 3. CHRONOLOGICAL RACE-BASED SPLIT
# ==========================================

races = (
    df[["race_id", "race_date"]]
    .drop_duplicates(subset=["race_id"])
    .sort_values(["race_date", "race_id"])
    .reset_index(drop=True)
)

split_index = int(len(races) * 0.8)

training_races = races.iloc[:split_index]["race_id"]
testing_races = races.iloc[split_index:]["race_id"]

train_mask = df["race_id"].isin(training_races)
test_mask = df["race_id"].isin(testing_races)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

print("\n========== TIME-BASED SPLIT ==========")
print("Total races:", len(races))
print("Training races:", len(training_races))
print("Testing races:", len(testing_races))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining period:")
print(train_df["race_date"].min(), "to", train_df["race_date"].max())

print("\nTesting period:")
print(test_df["race_date"].min(), "to", test_df["race_date"].max())

overlap = set(training_races).intersection(set(testing_races))

print("\nRaces appearing in both sets:", len(overlap))

if len(overlap) > 0:
    raise ValueError("Race leakage detected!")


# ==========================================
# 4. PREPROCESSING
# ==========================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)


# ==========================================
# 5. BUILD XGBOOST MODEL
# ==========================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.04,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    eval_metric="mae",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ]
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\n========== TRAINING MODEL ==========")

model.fit(X_train, y_train)

print("XGBoost training completed!")


# ==========================================
# 7. MAKE RAW PREDICTIONS
# ==========================================

raw_predictions = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")

mae = mean_absolute_error(y_test, raw_predictions)
rmse = np.sqrt(mean_squared_error(y_test, raw_predictions))
r2 = r2_score(y_test, raw_predictions)

print(f"XGBoost MAE:  {mae:.3f}")
print(f"XGBoost RMSE: {rmse:.3f}")
print(f"XGBoost R²:   {r2:.3f}")


# ==========================================
# 8. GRID POSITION BASELINE
# ==========================================

baseline_predictions = test_df["grid"].astype(float)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")
print(f"Grid position baseline MAE: {baseline_mae:.3f}")
print(f"XGBoost MAE:                {mae:.3f}")

if mae < baseline_mae:
    print("XGBoost performed better than the grid baseline.")
else:
    print("XGBoost did not outperform the grid baseline.")


# ==========================================
# 9. CREATE UNIQUE RACE PREDICTIONS
# ==========================================

results = test_df[
    [
        "season",
        "round",
        "race",
        "race_date",
        "race_id",
        "driver",
        "constructor",
        "grid",
        "position"
    ]
].copy()

# Preserve original decimal predictions
results["predicted_position_raw"] = raw_predictions

# Assign unique predicted finishing positions per race
results["predicted_position"] = (
    results.groupby("race_id")["predicted_position_raw"]
    .rank(method="first")
    .astype(int)
)

# Calculate error using the final ranked positions
results["prediction_error"] = (
    results["position"] - results["predicted_position"]
).abs()

results = results.sort_values(
    ["race_date", "predicted_position"]
)

results.to_csv("xgboost_test_predictions.csv", index=False)

print("\nTest predictions saved as xgboost_test_predictions.csv")


# ==========================================
# 10. RANKED PREDICTION EVALUATION
# ==========================================

ranked_mae = mean_absolute_error(
    results["position"],
    results["predicted_position"]
)

print("\n========== RANKED PREDICTION EVALUATION ==========")
print(f"Raw prediction MAE:    {mae:.3f}")
print(f"Ranked prediction MAE: {ranked_mae:.3f}")

duplicate_count = results.duplicated(
    ["race_id", "predicted_position"]
).sum()

print("Duplicate predicted positions:", duplicate_count)

if duplicate_count == 0:
    print("All races have unique predicted finishing positions!")


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

trained_preprocessor = model.named_steps["preprocessor"]
trained_regressor = model.named_steps["regressor"]

feature_names = trained_preprocessor.get_feature_names_out()
importances = trained_regressor.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance_df.head(15).to_string(index=False))


# ==========================================
# 12. SAVE MODEL AND METADATA
# ==========================================

joblib.dump(model, "f1_xgboost_model.joblib")

metadata = {
    "feature_columns": feature_columns,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "target": "position",
    "training_races": len(training_races),
    "testing_races": len(testing_races),
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "ranked_mae": ranked_mae,
    "grid_baseline_mae": baseline_mae
}

joblib.dump(metadata, "f1_xgboost_metadata.joblib")

print("\n========== SAVING COMPLETED ==========")
print("Model saved: f1_xgboost_model.joblib")
print("Metadata saved: f1_xgboost_metadata.joblib")
print("\nXGBoost pipeline completed successfully!")
