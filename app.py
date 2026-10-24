
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("traffic_volume_rf_small.joblib")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(f"Error details: {e}")
    st.stop()

# --------------------------------------------------
# 3. APP TITLE AND DESCRIPTION
# --------------------------------------------------
st.title("🚦 Traffic Volume Prediction")

st.write(
    "Predict hourly traffic volume using weather conditions "
    "and time-related information."
)

st.info(
    "Enter the weather and time details below, then click "
    "'Predict Traffic Volume' to estimate the number of vehicles."
)

st.divider()

# --------------------------------------------------
# 4. INPUT FORM
# --------------------------------------------------
st.subheader("Enter Prediction Details")

weather_options = [
    "Clear",
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Snow",
    "Squall",
    "Thunderstorm"
]

day_options = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

month_options = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

with st.form("traffic_prediction_form"):

    col1, col2 = st.columns(2)

    # Weather inputs
    with col1:
        st.markdown("### 🌦️ Weather Information")

        temp = st.number_input(
            "Temperature (Kelvin)",
            min_value=0.0,
            max_value=350.0,
            value=290.0,
            step=1.0
        )

        rain_1h = st.number_input(
            "Rainfall in the last hour (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        snow_1h = st.number_input(
            "Snowfall in the last hour (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        clouds_all = st.slider(
            "Cloud cover (%)",
            min_value=0,
            max_value=100,
            value=40
        )

        weather_main = st.selectbox(
            "Main weather condition",
            weather_options
        )

    # Time inputs
    with col2:
        st.markdown("### 🕒 Time Information")

        day_name = st.selectbox(
            "Day of the week",
            day_options
        )

        hour = st.slider(
            "Hour of the day",
            min_value=0,
            max_value=23,
            value=8,
            help="0 = midnight, 12 = noon, 23 = 11 PM"
        )

        month_name = st.selectbox(
            "Month",
            list(month_options.keys()),
            index=5
        )

        is_holiday_choice = st.selectbox(
            "Is it a holiday?",
            ["No", "Yes"]
        )

    # Derive features using the selected inputs
    day_of_week = day_mapping[day_name]
    month = month_options[month_name]

    is_weekend = int(
        day_name in ["Saturday", "Sunday"]
    )

    is_rush_hour = int(
        day_name not in ["Saturday", "Sunday"]
        and hour in [7, 8, 9, 16, 17, 18]
    )

    is_holiday = int(
        is_holiday_choice == "Yes"
    )

    submitted = st.form_submit_button(
        "🚦 Predict Traffic Volume",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. DISPLAY CALCULATED FEATURES AND PREDICTION
# --------------------------------------------------
if submitted:

    st.divider()
    st.subheader("Calculated Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.metric(
            "Weekend",
            "Yes" if is_weekend else "No"
        )

    with feature_col2:
        st.metric(
            "Rush Hour",
            "Yes" if is_rush_hour else "No"
        )

    with feature_col3:
        st.metric(
            "Holiday",
            "Yes" if is_holiday else "No"
        )

    # IMPORTANT:
    # Feature names and their meanings must match training.
    input_data = pd.DataFrame(
        [
            {
                "temp": temp,
                "rain_1h": rain_1h,
                "snow_1h": snow_1h,
                "clouds_all": clouds_all,
                "weather_main": weather_main,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "is_holiday": is_holiday
            }
        ]
    )

    try:
        prediction = float(model.predict(input_data)[0])

        # Traffic volume cannot be negative.
        prediction = max(0, prediction)

        st.divider()
        st.subheader("Prediction Result")

        st.success("Prediction completed successfully!")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Predicted Hourly Traffic Volume",
                value=f"{prediction:,.0f} vehicles"
            )

        with result_col2:
            st.metric(
                label="Prediction Period",
                value="1 hour"
            )

        st.caption(
            "This is a machine-learning estimate, not a live "
            "traffic count. Results depend on the training data "
            "and the information entered."
        )

        with st.expander("View the input data sent to the model"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error("Prediction failed. Please check the model and inputs.")
        st.exception(e)

# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Academic project: Traffic Volume Prediction Using Machine Learning"
)
