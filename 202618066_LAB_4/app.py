import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "model_artifacts"
DATA_DIR = BASE_DIR / "Data"


# ============================================================
# MODEL INFORMATION
# ============================================================

# Final model was trained using listings within approximately
# the 99th percentile of the Airbnb price distribution.
PRICE_RANGE_LIMIT = 799

REFERENCE_DATE = pd.Timestamp("2019-07-08")


# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        ARTIFACT_DIR / "model_99.pkl"
    )

    kmeans = joblib.load(
        ARTIFACT_DIR / "kmeans_99.pkl"
    )

    neighbourhood_target_map = joblib.load(
        ARTIFACT_DIR / "neigh_map_99.pkl"
    )

    room_neigh_target_map = joblib.load(
        ARTIFACT_DIR / "room_neigh_map_99.pkl"
    )

    neighbourhood_count_map = joblib.load(
        ARTIFACT_DIR / "neigh_counts_99.pkl"
    )

    global_mean = joblib.load(
        ARTIFACT_DIR / "global_mean_99.pkl"
    )

    return (
        model,
        kmeans,
        neighbourhood_target_map,
        room_neigh_target_map,
        neighbourhood_count_map,
        global_mean
    )


try:

    (
        model,
        kmeans,
        neighbourhood_target_map,
        room_neigh_target_map,
        neighbourhood_count_map,
        global_mean
    ) = load_artifacts()

except Exception as e:

    st.error(
        "Could not load the saved model artifacts."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LOAD NEIGHBOURHOOD OPTIONS
# ============================================================

@st.cache_data
def load_location_options():

    original_file = DATA_DIR / "AB_NYC_2019.csv"

    if original_file.exists():

        data = pd.read_csv(original_file)

        location_data = (
            data[
                [
                    "neighbourhood_group",
                    "neighbourhood"
                ]
            ]
            .dropna()
            .drop_duplicates()
            .sort_values(
                [
                    "neighbourhood_group",
                    "neighbourhood"
                ]
            )
        )

        location_map = {}

        for group in location_data[
            "neighbourhood_group"
        ].unique():

            neighbourhoods = (
                location_data.loc[
                    location_data[
                        "neighbourhood_group"
                    ] == group,
                    "neighbourhood"
                ]
                .sort_values()
                .tolist()
            )

            location_map[group] = neighbourhoods

        return location_map

    # --------------------------------------------------------
    # Fallback if original CSV is not available
    # --------------------------------------------------------

    return {
        "Bronx": [],
        "Brooklyn": [],
        "Manhattan": [],
        "Queens": [],
        "Staten Island": []
    }


location_map = load_location_options()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_geo_cluster(latitude, longitude):

    """
    Recreate the same KMeans geographic cluster used
    during model training.
    """

    # If KMeans was trained using a DataFrame with column names
    if hasattr(kmeans, "feature_names_in_"):

        geo_input = pd.DataFrame(
            {
                "latitude": [latitude],
                "longitude": [longitude]
            }
        )

    else:

        geo_input = np.array(
            [[latitude, longitude]]
        )

    cluster = kmeans.predict(geo_input)[0]

    return int(cluster)


def get_neighbourhood_encoding(neighbourhood):

    """
    Get target encoding for neighbourhood.
    For an unseen neighbourhood, use the global mean.
    """

    return float(
        neighbourhood_target_map.get(
            neighbourhood,
            global_mean
        )
    )


def get_room_neigh_encoding(
    room_type,
    neighbourhood_group
):

    """
    Get target encoding for:
    room_type × neighbourhood_group
    """

    key = (
        room_type,
        neighbourhood_group
    )

    value = room_neigh_target_map.get(
        key,
        global_mean
    )

    return float(value)


def get_neighbourhood_count(neighbourhood):

    """
    Number of listings belonging to the neighbourhood
    in the training data.
    """

    return float(
        neighbourhood_count_map.get(
            neighbourhood,
            1
        )
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🏠 NYC Airbnb Price Predictor")

st.markdown(
    """
    Estimate the nightly price of a New York City Airbnb
    listing using a tuned **LightGBM regression model**.

    The model was developed using the
    **AB_NYC_2019 Airbnb dataset** and combines geographic,
    property, review, host, availability, and engineered
    neighbourhood features.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        """
        **Model:** LightGBM Regressor

        **Target transformation:** `log1p(price)`

        **Price range:** Approximately up to the
        99th percentile of the training distribution.

        **Approximate upper modelling range:** $799/night
        """
    )

    st.info(
        """
        This model was trained using historical NYC Airbnb
        data from 2019. Predictions should be interpreted
        as estimates rather than exact current market prices.
        """
    )


# ============================================================
# INPUT FORM
# ============================================================

st.header("Enter Listing Details")

with st.form("airbnb_prediction_form"):

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader("📍 Location")

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group / Borough",
        [
            "Bronx",
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Staten Island"
        ]
    )


    neighbourhood_options = location_map.get(
        neighbourhood_group,
        []
    )


    # Fallback to neighbourhoods from target encoding map
    if len(neighbourhood_options) == 0:

        neighbourhood_options = sorted(
            [
                str(x)
                for x in neighbourhood_target_map.keys()
            ]
        )


    neighbourhood = st.selectbox(
        "Neighbourhood",
        neighbourhood_options
    )


    col1, col2 = st.columns(2)


    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=40.49,
            max_value=40.92,
            value=40.7580,
            step=0.001,
            format="%.6f"
        )


    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=-74.25,
            max_value=-73.70,
            value=-73.9855,
            step=0.001,
            format="%.6f"
        )


    # --------------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------------

    st.subheader("🏡 Property Details")


    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )


    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=365,
        value=3,
        step=1
    )


    # --------------------------------------------------------
    # REVIEW DETAILS
    # --------------------------------------------------------

    st.subheader("⭐ Review Information")


    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=1000,
        value=10,
        step=1
    )


    if number_of_reviews == 0:

        reviews_per_month = 0.0
        has_review = 0
        days_since_last_review = -1

        st.info(
            """
            Because the listing has no reviews:

            • Reviews per month = 0  
            • Has review = 0  
            • Days since last review = -1
            """
        )

    else:

        has_review = 1


        reviews_per_month = st.number_input(
            "Reviews Per Month",
            min_value=0.0,
            max_value=60.0,
            value=1.0,
            step=0.1
        )


        days_since_last_review = st.number_input(
            "Days Since Last Review",
            min_value=0,
            max_value=3000,
            value=30,
            step=1,
            help=(
                "The original model calculated review recency "
                "relative to the historical dataset reference "
                "date of 2019-07-08."
            )
        )


    # --------------------------------------------------------
    # HOST INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Host Information")


    calculated_host_listings_count = st.number_input(
        "Number of Listings Managed by Host",
        min_value=1,
        max_value=500,
        value=1,
        step=1
    )


    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    st.subheader("📅 Availability")


    availability_365 = st.slider(
        "Days Available Per Year",
        min_value=0,
        max_value=365,
        value=180
    )


    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "Predict Nightly Price",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        # ----------------------------------------------------
        # GEO CLUSTER
        # ----------------------------------------------------

        geo_cluster = predict_geo_cluster(
            latitude,
            longitude
        )


        # ----------------------------------------------------
        # TARGET ENCODING
        # ----------------------------------------------------

        neighbourhood_target_enc = (
            get_neighbourhood_encoding(
                neighbourhood
            )
        )


        room_neigh_target_enc = (
            get_room_neigh_encoding(
                room_type,
                neighbourhood_group
            )
        )


        # ----------------------------------------------------
        # NEIGHBOURHOOD LISTING COUNT
        # ----------------------------------------------------

        neighbourhood_listing_count = (
            get_neighbourhood_count(
                neighbourhood
            )
        )


        # ----------------------------------------------------
        # CREATE FINAL MODEL INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "latitude": [
                    float(latitude)
                ],

                "longitude": [
                    float(longitude)
                ],

                "minimum_nights": [
                    int(minimum_nights)
                ],

                "number_of_reviews": [
                    int(number_of_reviews)
                ],

                "reviews_per_month": [
                    float(reviews_per_month)
                ],

                "calculated_host_listings_count": [
                    int(
                        calculated_host_listings_count
                    )
                ],

                "availability_365": [
                    int(availability_365)
                ],

                "has_review": [
                    int(has_review)
                ],

                "days_since_last_review": [
                    int(days_since_last_review)
                ],

                "geo_cluster": [
                    int(geo_cluster)
                ],

                "neighbourhood_target_enc": [
                    float(
                        neighbourhood_target_enc
                    )
                ],

                "room_neigh_target_enc": [
                    float(
                        room_neigh_target_enc
                    )
                ],

                "neighbourhood_listing_count": [
                    float(
                        neighbourhood_listing_count
                    )
                ],

                "neighbourhood_group": [
                    neighbourhood_group
                ],

                "room_type": [
                    room_type
                ]
            }
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        predicted_price = model.predict(
            input_data
        )[0]


        predicted_price = float(
            predicted_price
        )


        # Avoid displaying impossible negative prices
        predicted_price = max(
            predicted_price,
            0
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("💰 Estimated Nightly Price")


        st.metric(
            label="Predicted Airbnb Price",
            value=f"${predicted_price:,.2f} / night"
        )


        if predicted_price > PRICE_RANGE_LIMIT:

            st.warning(
                f"""
                The predicted value is above approximately
                ${PRICE_RANGE_LIMIT}, which is outside the
                primary price range used to train the final model.

                This prediction should therefore be interpreted
                with additional caution.
                """
            )

        else:

            st.success(
                """
                The prediction falls within the primary price
                range represented by the final model.
                """
            )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("Listing Summary")


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"**Borough:** {neighbourhood_group}"
            )

            st.write(
                f"**Neighbourhood:** {neighbourhood}"
            )

            st.write(
                f"**Room Type:** {room_type}"
            )

            st.write(
                f"**Minimum Nights:** {minimum_nights}"
            )


        with summary_col2:

            st.write(
                f"**Reviews:** {number_of_reviews}"
            )

            st.write(
                f"**Availability:** {availability_365} days"
            )

            st.write(
                f"**Host Listings:** "
                f"{calculated_host_listings_count}"
            )

            st.write(
                f"**Geo Cluster:** {geo_cluster}"
            )


        # ----------------------------------------------------
        # ENGINEERED FEATURES
        # ----------------------------------------------------

        with st.expander(
            "View Engineered Model Features"
        ):

            st.write(
                "**Geographic Cluster:**",
                geo_cluster
            )

            st.write(
                "**Neighbourhood Target Encoding:**",
                round(
                    neighbourhood_target_enc,
                    2
                )
            )

            st.write(
                "**Room × Borough Target Encoding:**",
                round(
                    room_neigh_target_enc,
                    2
                )
            )

            st.write(
                "**Neighbourhood Listing Count:**",
                int(
                    neighbourhood_listing_count
                )
            )

            st.write(
                "**Has Review:**",
                has_review
            )

            st.write(
                "**Days Since Last Review:**",
                days_since_last_review
            )


        # ----------------------------------------------------
        # OPTIONAL MODEL INPUT TABLE
        # ----------------------------------------------------

        with st.expander(
            "View Final Model Input"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("📊 Final Model Performance")

st.markdown(
    """
    The final model was selected after comparing multiple
    regression algorithms, feature-engineering strategies,
    and price-range treatments.

    **Final 99th-percentile model results:**

    | Dataset | R² |
    |---|---:|
    | Validation | **0.4838** |
    | Final unseen test set | **0.5145** |

    The 99th-percentile model was selected as a compromise
    between predictive performance and coverage of the Airbnb
    price range.
    """
)


# ============================================================
# MODEL LIMITATIONS
# ============================================================

st.header("⚠️ Important Limitations")

st.markdown(
    f"""
    - The model was trained using **New York City Airbnb data
      from 2019**.
    - The final model focuses primarily on listings within
      approximately the **99th percentile of the historical
      price distribution**, around **${PRICE_RANGE_LIMIT} per
      night**.
    - Extremely expensive luxury properties may be predicted
      less accurately.
    - Airbnb market conditions, neighbourhood demand, and
      prices may have changed since 2019.
    - The prediction should be interpreted as an **estimated
      nightly price**, not a guaranteed market price.
    """
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.caption(
    """
    DS605 – Fundamentals of Machine Learning |
    Lab 4 – End-to-End Airbnb Price Prediction
    """
)