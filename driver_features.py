
import pandas as pd

# ==========================================
# 1. LOAD AND PREPARE DATASET
# ==========================================

data = pd.read_csv("f1_results_2022_2025_cleaned.csv")

data["race_date"] = pd.to_datetime(data["race_date"])

data["race_id"] = (
    data["season"].astype(str)
    + "_"
    + data["round"].astype(str)
)

# ==========================================
# 2. DRIVER HISTORICAL FEATURES
# ==========================================

data = data.sort_values(
    ["driver_id", "race_date", "season", "round"]
).reset_index(drop=True)

data["driver_avg_previous"] = (
    data.groupby("driver_id")["position"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

data["driver_avg_last_5"] = (
    data.groupby("driver_id")["position"]
    .transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)

data["driver_previous_races"] = (
    data.groupby("driver_id").cumcount()
)

# ==========================================
# 3. CONSTRUCTOR HISTORICAL FEATURES
# ==========================================

constructor_races = (
    data.groupby(
        ["constructor_id", "race_id", "race_date"]
    )["position"]
    .mean()
    .reset_index(name="constructor_race_avg")
)

constructor_races = constructor_races.sort_values(
    ["constructor_id", "race_date"]
)

constructor_races["constructor_avg_previous"] = (
    constructor_races.groupby("constructor_id")[
        "constructor_race_avg"
    ].transform(lambda s: s.shift(1).expanding().mean())
)

constructor_races["constructor_avg_last_5"] = (
    constructor_races.groupby("constructor_id")[
        "constructor_race_avg"
    ].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)

data = data.merge(
    constructor_races[
        [
            "constructor_id",
            "race_id",
            "constructor_avg_previous",
            "constructor_avg_last_5"
        ]
    ],
    on=["constructor_id", "race_id"],
    how="left"
)

# ==========================================
# 4. DRIVER CIRCUIT-SPECIFIC PERFORMANCE
# ==========================================

data = data.sort_values(
    ["driver_id", "circuit_id", "race_date", "season", "round"]
).reset_index(drop=True)

data["driver_circuit_avg_previous"] = (
    data.groupby(["driver_id", "circuit_id"])["position"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

data["driver_circuit_previous_races"] = (
    data.groupby(["driver_id", "circuit_id"]).cumcount()
)

# ==========================================
# 5. CIRCUIT LENGTH FEATURE
# ==========================================

circuit_lengths = {
    "albert_park": 5.278,
    "imola": 4.909,
    "monza": 5.793,
    "rodriguez": 4.304,
    "interlagos": 4.309,
    "bahrain": 5.412,
    "baku": 6.003,
    "villeneuve": 4.361,
    "zandvoort": 4.259,
    "ricard": 5.842,
    "catalunya": 4.657,
    "monaco": 3.337,
    "spa": 7.004,
    "americas": 5.513,
    "hungaroring": 4.381,
    "jeddah": 6.174,
    "vegas": 6.201,
    "losail": 5.419,
    "marina_bay": 4.940,
    "miami": 5.412,
    "red_bull_ring": 4.318,
    "shanghai": 5.451,
    "silverstone": 5.891,
    "suzuka": 5.807,
    "yas_marina": 5.281
}

# Match circuit IDs to their lap lengths
data["circuit_length_km"] = data["circuit_id"].map(
    circuit_lengths
)

# Barcelona used a different layout in 2022
data.loc[
    (data["circuit_id"] == "catalunya")
    & (data["season"] == 2022),
    "circuit_length_km"
] = 4.675

# Check that every circuit has a length
missing_circuits = data.loc[
    data["circuit_length_km"].isna(),
    "circuit_id"
].unique()

if len(missing_circuits) > 0:
    raise ValueError(
        f"Missing circuit lengths for: {missing_circuits}"
    )

# ==========================================
# 6. SAVE FINAL FEATURE DATASET
# ==========================================

data = data.sort_values(
    ["race_date", "season", "round", "driver_id"]
).reset_index(drop=True)

data.to_csv("f1_driver_features.csv", index=False)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("========== F1 FEATURE ENGINEERING ==========")

print("\nDataset shape:", data.shape)

print("\nCircuit lengths:")
print(
    data[["circuit_id", "circuit", "circuit_length_km"]]
    .drop_duplicates()
    .sort_values("circuit")
    .to_string(index=False)
)

print("\nMissing values in historical features:")
print(data[[
    "driver_avg_previous",
    "driver_avg_last_5",
    "constructor_avg_previous",
    "constructor_avg_last_5",
    "driver_circuit_avg_previous"
]].isnull().sum())

print("\nMissing circuit lengths:",
      data["circuit_length_km"].isnull().sum())

print("\nAll engineered features:")
print([
    "driver_avg_previous",
    "driver_avg_last_5",
    "driver_previous_races",
    "constructor_avg_previous",
    "constructor_avg_last_5",
    "driver_circuit_avg_previous",
    "driver_circuit_previous_races",
    "circuit_length_km"
])

print("\nFeature engineering completed successfully!")
print("Dataset saved as f1_driver_features.csv")
