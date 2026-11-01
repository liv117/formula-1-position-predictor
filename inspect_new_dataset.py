
import pandas as pd

data = pd.read_csv("f1_results_2022_2025_raw.csv")

# 1. Find races with fewer or more than 20 records
print("\n========== RACE RECORD COUNTS ==========")

race_counts = (
    data.groupby(["season", "round", "race"])
    .size()
    .reset_index(name="driver_records")
)

print(race_counts[race_counts["driver_records"] != 20].to_string(index=False))


# 2. Check finishing positions
print("\n========== FINISHING POSITION DISTRIBUTION ==========")

print(
    data.groupby("season")["position"]
    .describe()
)


# 3. Check grid positions
print("\n========== GRID POSITION CHECK ==========")

print("Minimum grid:", data["grid"].min())
print("Maximum grid:", data["grid"].max())

print("\nRecords with grid position 0:")
print(data[data["grid"] == 0][
    ["season", "race", "driver", "grid", "position", "status"]
].to_string(index=False))


# 4. Check driver race statuses
print("\n========== RACE STATUS DISTRIBUTION ==========")

print(
    data.groupby(["season", "status"])
    .size()
    .to_string()
)


# 5. Check non-numeric finishing position labels
print("\n========== POSITION TEXT VALUES ==========")

print(data["position_text"].value_counts().to_string())


# 6. Check number of unique drivers per season
print("\n========== UNIQUE DRIVERS ==========")

print(data.groupby("season")["driver_id"].nunique())


# 7. Check number of unique constructors per season
print("\n========== UNIQUE CONSTRUCTORS ==========")

print(data.groupby("season")["constructor_id"].nunique())




print("\n========== DATASET COLUMNS ==========")

print(data.columns.tolist())

print("\n========== DATA TYPES ==========")

print(data.dtypes)

print("\n========== FIRST 5 RECORDS ==========")

print(data.head().to_string())

print("\n========== UNIQUE VALUES ==========")

for column in data.columns:
    print(f"\n{column}: {data[column].nunique()} unique values")


    print("\n========== POSITION AND POSITION TEXT ==========")

print(
    data.groupby(
        ["position_text", "position"]
    ).size().to_string()
)



print("\n========== WITHDRAWN / DID NOT START ==========")

special = data[
    data["status"].str.contains(
        "withdrew|did not start",
        case=False,
        na=False
    )
]

print(
    special[
        ["season", "race", "driver", "grid",
         "position", "position_text", "status"]
    ].to_string(index=False)
)