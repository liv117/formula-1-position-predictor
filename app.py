
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

from pathlib import Path


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="F1 Position Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. FILE CONFIGURATION
# ==================================================

DATA_FILE = "f1_driver_features.csv"
MODEL_FILE = "f1_xgboost_model.joblib"
METADATA_FILE = "f1_xgboost_metadata.joblib"

IMAGE_FILE = Path("assets/f1_homepage.jpg.jpeg")

TRAINING_END_DATE = pd.Timestamp("2025-04-20")


# ==================================================
# 3. LOAD HOMEPAGE IMAGE
# ==================================================

def get_homepage_image():

    if not IMAGE_FILE.exists():
        return None

    with open(IMAGE_FILE, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    return encoded_image


# ==================================================
# 4. CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background: #090a0d;
    color: #ffffff;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #ffffff;
    font-family: Arial, sans-serif;
}

p, label, span {
    color: #d0d0d0;
}

.nav-brand {
    color: #ffffff;
    font-size: 21px;
    font-weight: 900;
    letter-spacing: 2px;
}

.nav-brand span {
    color: #e10600;
}

.nav-detail {
    color: #888888;
    font-size: 11px;
    letter-spacing: 3px;
    text-align: right;
    padding-top: 8px;
}

.hero {
    position: relative;
    width: 100%;
    height: 620px;
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-color: #111115;
    border: 1px solid #29292e;
    border-radius: 4px;
    overflow: hidden;
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        180deg,
        rgba(0,0,0,0.02) 45%,
        rgba(0,0,0,0.12) 70%,
        rgba(0,0,0,0.72) 100%
    );
}

.hero-label {
    position: absolute;
    bottom: 35px;
    left: 38px;
    color: #ffffff;
    font-size: 12px;
    letter-spacing: 4px;
    font-weight: 700;
}

.hero-label span {
    color: #e10600;
}

/* NEW F1 TAGLINE */

.f1-tagline {
    text-align: center;
    color: #d0d0d0;
    font-size: 19px;
    font-style: italic;
    font-weight: 400;
    letter-spacing: 1.5px;
    margin-top: 22px;
    margin-bottom: 8px;
}

.info-card {
    background: #121318;
    border: 1px solid #292a30;
    border-top: 2px solid #e10600;
    padding: 24px;
    min-height: 145px;
    margin-top: 18px;
}

.card-label {
    color: #e10600;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 15px;
}

.card-title {
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 8px;
}

.card-description {
    color: #999ba3;
    font-size: 13px;
    line-height: 1.6;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #e10600;
    color: white;
    border: 1px solid #e10600;
    border-radius: 2px;
    font-weight: 800;
    letter-spacing: 1.5px;
    padding: 13px 22px;
    min-height: 48px;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #ff2922;
    border-color: #ff2922;
    color: white;
}

div[data-baseweb="select"] > div {
    background: #17181e;
    border-color: #363740;
    color: white;
}

div[data-testid="stNumberInput"] input {
    background: #17181e;
    color: white;
    border: 1px solid #363740;
}

div[data-testid="stMetric"] {
    background: #15161b;
    padding: 16px;
    border-left: 2px solid #e10600;
}

.result-card {
    background: #17181e;
    border: 1px solid #303139;
    border-top: 3px solid #e10600;
    padding: 22px;
    border-radius: 3px;
    margin-bottom: 8px;
    text-align: center;
}

.position-number {
    color: #e10600;
    font-size: 34px;
    font-weight: 900;
}

.driver-name {
    color: white;
    font-size: 19px;
    font-weight: bold;
}

.constructor-name {
    color: #aaaaaa;
    font-size: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #303139;
}

.footer {
    text-align: center;
    color: #65666d;
    font-size: 10px;
    letter-spacing: 2px;
    padding: 25px 0 10px 0;
}

.footer span {
    color: #e10600;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 5. LOAD DATASET
# ==================================================

@st.cache_data
def load_data():

    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "season",
        "round",
        "race",
        "race_id",
        "race_date",
        "driver",
        "driver_id",
        "constructor",
        "constructor_id",
        "circuit_id",
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

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    data["race_date"] = pd.to_datetime(
        data["race_date"]
    )

    numeric_columns = [
        "season",
        "round",
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

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "season",
            "round",
            "race_date",
            "race_id",
            "driver",
            "constructor",
            "circuit_length_km"
        ]
    )

    return data


# ==================================================
# 6. LOAD XGBOOST MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} was not found."
        )

    if not Path(METADATA_FILE).exists():
        raise FileNotFoundError(
            f"{METADATA_FILE} was not found."
        )

    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)

    return model, metadata


try:
    data = load_data()
    model, metadata = load_model()

except Exception as error:
    st.error(f"Application loading error: {error}")
    st.stop()


feature_columns = metadata["feature_columns"]


# ==================================================
# 7. CHECK MODEL FEATURES
# ==================================================

if "circuit_length_km" not in feature_columns:
    st.error(
        "Circuit length is missing from the model's "
        "training features. Please check the model metadata."
    )
    st.stop()


missing_features = [
    column for column in feature_columns
    if column not in data.columns
]

if missing_features:
    st.error(
        f"The dataset is missing model features: {missing_features}"
    )
    st.stop()


# ==================================================
# 8. PREPARE HELD-OUT RACES
# ==================================================

demo_data = data[
    data["race_date"] > TRAINING_END_DATE
].copy()

if demo_data.empty:
    st.error(
        "No held-out races were found after the model's "
        "training period."
    )
    st.stop()


# ==================================================
# 9. SESSION STATE
# ==================================================

if "screen" not in st.session_state:
    st.session_state.screen = 0

if "selected_season" not in st.session_state:
    st.session_state.selected_season = None

if "selected_race" not in st.session_state:
    st.session_state.selected_race = None

if "selected_round" not in st.session_state:
    st.session_state.selected_round = None

if "selected_circuit" not in st.session_state:
    st.session_state.selected_circuit = None

if "selected_circuit_length" not in st.session_state:
    st.session_state.selected_circuit_length = None

if "drivers" not in st.session_state:
    st.session_state.drivers = None

if "grid_positions" not in st.session_state:
    st.session_state.grid_positions = {}

if "predictions" not in st.session_state:
    st.session_state.predictions = None


# ==================================================
# 10. BRIGHTER BACKGROUND FOR SCREENS 1, 2 AND 3
# ==================================================

if st.session_state.screen in [1, 2, 3]:

    background_image = get_homepage_image()

    if background_image is not None:

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(7, 8, 12, 0.65),
                        rgba(7, 8, 12, 0.72)
                    ),
                    url(
                        "data:image/jpeg;base64,{background_image}"
                    ) !important;

                background-size: cover !important;
                background-position: center center !important;
                background-attachment: fixed !important;
                background-repeat: no-repeat !important;
                background-color: #090a0d !important;
            }}

            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            .block-container {{
                background: transparent !important;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# 11. TOP NAVIGATION
# ==================================================

nav1, nav2 = st.columns([2, 1])

with nav1:
    st.markdown(
        '<div class="nav-brand">'
        'F1 <span>POSITION</span> PREDICTOR'
        '</div>',
        unsafe_allow_html=True
    )

with nav2:
    st.markdown(
        '<div class="nav-detail">'
        'MACHINE LEARNING / RACE ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown(
    "<hr style='border:1px solid #24252b; margin-top:15px;'>",
    unsafe_allow_html=True
)


# ==================================================
# 12. SCREEN 0: HOMEPAGE
# ==================================================

if st.session_state.screen == 0:

    homepage_image = get_homepage_image()

    if homepage_image is None:

        st.error(
            "Homepage image not found. Please check that "
            "assets/f1_homepage.jpg.jpeg exists."
        )

        st.info(
            "Your project should contain an assets folder "
            "with f1_homepage.jpg.jpeg inside it."
        )

    else:

        st.markdown(
            f"""
            <div class="hero"
                style="
                    background-image:
                    linear-gradient(
                        180deg,
                        rgba(0,0,0,0.02),
                        rgba(0,0,0,0.12)
                    ),
                    url('data:image/jpeg;base64,{homepage_image}');
                ">
                <div class="hero-overlay"></div>
                <div class="hero-label">
                    <span>01</span> / THE GRID.
                    THE DATA. THE PREDICTION.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # NEW F1 TAGLINE

    st.markdown(
        """
        <div class="f1-tagline">
            It's lights out and away we go.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    card1, card2 = st.columns(2)

    with card1:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">01 / NEXT RACE</div>
                <div class="card-title">Choose Your Grand Prix</div>
                <div class="card-description">
                    Select a season and circuit to explore
                    predicted Formula 1 finishing positions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-label">02 / LATEST UPDATE</div>
                <div class="card-title">Powered by XGBoost</div>
                <div class="card-description">
                    Driver history, constructor performance,
                    starting grid and circuit characteristics
                    come together to generate predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _, button_column, _ = st.columns([1, 2, 1])

    with button_column:

        if st.button(
            "ENTER THE PREDICTOR →",
            use_container_width=True
        ):

            st.session_state.screen = 1
            st.rerun()


# ==================================================
# 13. SCREEN 1: SEASON AND RACE SELECTION
# ==================================================

elif st.session_state.screen == 1:

    st.subheader("01 / SELECT YOUR RACE")

    st.write(
        "Choose an F1 season and Grand Prix to generate "
        "a predicted finishing order."
    )

    st.info(
        "This demonstration uses races after 20 April 2025, "
        "which were not included in the model's training data."
    )

    seasons = sorted(
        demo_data["season"].unique().astype(int).tolist(),
        reverse=True
    )

    season_index = (
        seasons.index(st.session_state.selected_season)
        if st.session_state.selected_season in seasons
        else 0
    )

    season = st.selectbox(
        "Select Season",
        seasons,
        index=season_index
    )

    season_data = demo_data[
        demo_data["season"] == season
    ]

    races = (
        season_data[
            ["round", "race", "race_id"]
        ]
        .drop_duplicates(subset=["race_id"])
        .sort_values("round")
        .reset_index(drop=True)
    )

    race_lookup = {}

    for _, row in races.iterrows():

        label = (
            f"Round {int(row['round'])} — {row['race']}"
        )

        race_lookup[label] = row

    race_labels = list(race_lookup.keys())

    selected_option = st.selectbox(
        "Select Grand Prix",
        race_labels
    )

    selected_race_row = race_lookup[selected_option]

    selected_race = selected_race_row["race"]
    selected_round = int(selected_race_row["round"])
    selected_race_id = selected_race_row["race_id"]

    race_data = season_data[
        season_data["race_id"] == selected_race_id
    ].copy()

    selected_circuit = (
        race_data["circuit"].iloc[0]
        if "circuit" in race_data.columns
        else race_data["circuit_id"].iloc[0]
    )

    selected_circuit_length = float(
        race_data["circuit_length_km"].iloc[0]
    )

    st.markdown("### Selected Race")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Season", season)

    with col2:
        st.metric("Race Round", selected_round)

    with col3:
        st.metric(
            "Circuit Length",
            f"{selected_circuit_length:.3f} km"
        )

    st.write(f"**Grand Prix:** {selected_race}")
    st.write(f"**Circuit:** {selected_circuit}")

    if st.button(
        "CONTINUE →",
        use_container_width=True
    ):

        race_data = (
            race_data
            .drop_duplicates(subset=["driver_id"])
            .sort_values("grid")
            .reset_index(drop=True)
        )

        if len(race_data) > 20:
            st.error(
                "This race has more than 20 driver records."
            )
            st.stop()

        st.session_state.selected_season = season
        st.session_state.selected_race = selected_race
        st.session_state.selected_round = selected_round
        st.session_state.selected_circuit = selected_circuit
        st.session_state.selected_circuit_length = selected_circuit_length

        st.session_state.drivers = race_data

        used_positions = set()
        grid_defaults = {}

        for _, row in race_data.iterrows():

            driver = row["driver"]
            original_grid = row["grid"]

            if pd.notna(original_grid):
                original_grid = int(original_grid)
            else:
                original_grid = 0

            if (
                original_grid >= 1
                and original_grid <= 20
                and original_grid not in used_positions
            ):

                grid_defaults[driver] = original_grid
                used_positions.add(original_grid)

        available_positions = [
            position
            for position in range(1, 21)
            if position not in used_positions
        ]

        for _, row in race_data.iterrows():

            driver = row["driver"]

            if driver not in grid_defaults:

                position = available_positions.pop(0)
                grid_defaults[driver] = position

        st.session_state.grid_positions = grid_defaults
        st.session_state.predictions = None
        st.session_state.screen = 2

        st.rerun()

    if st.button("← HOMEPAGE"):
        st.session_state.screen = 0
        st.rerun()


# ==================================================
# 14. SCREEN 2: STARTING GRID
# ==================================================

elif st.session_state.screen == 2:

    st.subheader("02 / STARTING GRID")

    st.write(
        f"**{st.session_state.selected_season} "
        f"{st.session_state.selected_race}**"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Circuit:** {st.session_state.selected_circuit}"
        )

    with col2:
        st.write(
            f"**Track Length:** "
            f"{st.session_state.selected_circuit_length:.3f} km"
        )

    st.write(
        "Set the starting grid position for each driver."
    )

    drivers = st.session_state.drivers

    st.caption(
        f"{len(drivers)} participating drivers"
    )

    with st.form("grid_form"):

        grid_inputs = {}

        for index, driver_row in drivers.iterrows():

            driver = driver_row["driver"]
            constructor = driver_row["constructor"]

            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:
                st.markdown(f"**{driver}**")

            with col2:
                st.caption(constructor)

            with col3:

                grid_inputs[driver] = st.number_input(
                    "Grid",
                    min_value=1,
                    max_value=20,
                    value=int(
                        st.session_state.grid_positions[driver]
                    ),
                    key=f"grid_{driver}"
                )

        predict_button = st.form_submit_button(
            "GENERATE PREDICTIONS →",
            use_container_width=True
        )

    if st.button("← BACK TO RACE SELECTION"):
        st.session_state.screen = 1
        st.rerun()

    if predict_button:

        positions = list(grid_inputs.values())

        if len(positions) != len(set(positions)):

            st.error(
                "Each driver must have a different starting "
                "grid position. Please correct the duplicates."
            )

        else:

            st.session_state.grid_positions = grid_inputs

            prediction_input = drivers.copy()

            prediction_input["grid"] = (
                prediction_input["driver"].map(grid_inputs)
            )

            prediction_input["grid_is_zero"] = (
                prediction_input["grid"] == 0
            ).astype(int)

            prediction_input["circuit_length_km"] = (
                st.session_state.selected_circuit_length
            )

            if prediction_input["circuit_length_km"].isna().any():
                st.error(
                    "Circuit length is missing for this race."
                )
                st.stop()

            X_predict = prediction_input[
                feature_columns
            ].copy()

            raw_predictions = model.predict(X_predict)

            prediction_input["raw_prediction"] = (
                raw_predictions
            )

            prediction_input["predicted_position"] = (
                prediction_input["raw_prediction"]
                .rank(method="first")
                .astype(int)
            )

            prediction_input = (
                prediction_input
                .sort_values("predicted_position")
                .reset_index(drop=True)
            )

            st.session_state.predictions = prediction_input
            st.session_state.screen = 3

            st.rerun()


# ==================================================
# 15. SCREEN 3: PREDICTION RESULTS
# ==================================================

elif st.session_state.screen == 3:

    st.subheader("03 / PREDICTED RACE RESULTS")

    st.write(
        f"**{st.session_state.selected_season} "
        f"{st.session_state.selected_race}**"
    )

    st.write(
        f"**Circuit:** {st.session_state.selected_circuit} "
        f"({st.session_state.selected_circuit_length:.3f} km)"
    )

    results = st.session_state.predictions

    if results is None or results.empty:

        st.error("No predictions are available.")
        st.stop()

    if results["predicted_position"].duplicated().any():

        st.error(
            "Duplicate predicted positions detected."
        )
        st.stop()

    st.success(
        "Prediction generated successfully — "
        "every driver has a unique finishing position."
    )

    st.markdown("### 🏆 PREDICTED PODIUM")

    podium = results.head(3)

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for index, (_, row) in enumerate(podium.iterrows()):

        with columns[index]:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="position-number">
                        P{row['predicted_position']}
                    </div>
                    <div class="driver-name">
                        {row['driver']}
                    </div>
                    <div class="constructor-name">
                        {row['constructor']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("### FULL PREDICTED CLASSIFICATION")

    display_results = results[
        [
            "predicted_position",
            "driver",
            "constructor",
            "grid"
        ]
    ].copy()

    display_results.columns = [
        "Predicted Position",
        "Driver",
        "Constructor",
        "Starting Grid"
    ]

    st.dataframe(
        display_results,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Predictions are generated by an XGBoost regression "
        "model using starting grid position, driver history, "
        "constructor history and circuit characteristics, "
        "including circuit length."
    )

    st.caption(
        "The model produces estimated finishing scores, "
        "which are ranked to create a unique finishing order. "
        "Predictions are not guaranteed race outcomes."
    )

    csv_data = display_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="DOWNLOAD PREDICTED RESULTS",
        data=csv_data,
        file_name="f1_predicted_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "← CHANGE GRID POSITIONS",
            use_container_width=True
        ):

            st.session_state.screen = 2
            st.rerun()

    with col2:

        if st.button(
            "NEW RACE",
            use_container_width=True
        ):

            st.session_state.screen = 1
            st.session_state.predictions = None
            st.rerun()

    if st.button("← HOMEPAGE"):
        st.session_state.screen = 0
        st.rerun()


# ==================================================
# 16. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        F1 POSITION PREDICTOR
        <br><br>
        MACHINE LEARNING PROJECT
        <span> | </span>
        POWERED BY XGBOOST
        <br><br>
        PREDICTIONS ARE ESTIMATES, NOT GUARANTEED RESULTS.
    </div>
    """,
    unsafe_allow_html=True
)
