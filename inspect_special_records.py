import pandas as pd

data = pd.read_csv("f1_results_2022_2025_raw.csv")

print("\n========== RACES WITH FEWER THAN 20 RECORDS ==========")

race_counts = data.groupby(
    ["season", "round", "race"]
).size().reset_index(name="records")

incomplete_races = race_counts[race_counts["records"] < 20]

for _, race in incomplete_races.iterrows():

    print(f"\n{race['season']} - {race['race']}")

    race_data = data[
        (data["season"] == race["season"]) &
        (data["round"] == race["round"])
    ]

    print(race_data[
        ["driver", "constructor", "grid", "position", "status"]
    ].to_string(index=False))


print("\n========== ALL GRID ZERO RECORDS ==========")

print(data[data["grid"] == 0][
    ["season", "race", "driver", "constructor", "position", "status"]
].to_string(index=False))


print("\n========== NON-NUMERIC POSITION TEXT ==========")

print(data[data["position_text"].isin(["R", "W", "D"])][
    ["season", "race", "driver", "position", "position_text", "status"]
].to_string(index=False))