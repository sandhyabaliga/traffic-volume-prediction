# Traffic Volume Prediction Using Machine Learning

## Project Overview

Traffic Volume Prediction is a machine learning project that predicts hourly traffic volume using historical traffic records, weather conditions, and time-based features.

The project uses the **Metro Interstate Traffic Volume Dataset** from the UCI Machine Learning Repository. It includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and a Streamlit web application for interactive predictions.

## Objectives

- Analyze historical traffic volume patterns.
- Study the relationship between traffic volume, time, and weather conditions.
- Develop and compare multiple machine learning regression models.
- Evaluate models using MAE, RMSE, and R².
- Deploy a web application for user-friendly traffic volume predictions.

## Dataset

**Dataset:** Metro Interstate Traffic Volume  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

The original dataset contains 48,204 records and 9 columns.

Important attributes include:
- `date_time` — date and time of the traffic observation
- `traffic_volume` — recorded traffic volume
- `temp` — temperature
- `rain_1h` — rainfall during one hour
- `snow_1h` — snowfall during one hour
- `clouds_all` — cloud coverage
- `weather_main` — general weather condition
- `weather_description` — detailed weather description
- `holiday` — holiday information

## Methodology

### 1. Data Preprocessing
- Removed 17 exact duplicate records.
- Aggregated records with identical timestamps.
- Sorted observations chronologically.
- Used an 80:20 chronological train-test split.
- Applied median imputation to numerical features and most-frequent imputation to categorical features.
- Scaled numerical features and one-hot encoded categorical features within the model pipeline.

After timestamp aggregation, the dataset contained 40,575 records.

### 2. Feature Engineering

The model uses weather and time-based features, including:
- Hour of the day
- Day of the week
- Month
- Weekend indicator
- Rush-hour indicator
- Holiday indicator
- Temperature, rainfall, snowfall, cloud coverage, and general weather condition

### 3. Exploratory Data Analysis

The analysis examined traffic volume by hour, day of the week, month, and weather condition. Average traffic volume was highest around 4 PM and lowest around 3 AM in the analyzed dataset.

### 4. Machine Learning Models

The following regression approaches were evaluated:
- Baseline prediction using average traffic volume by day of the week and hour
- Linear Regression
- Ridge Regression
- K-Nearest Neighbors (KNN) Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Extra Trees Regression
- Feed-Forward Neural Network (MLP Regressor)

## Model Evaluation

The original experiments used a chronological test set and the following metrics:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline (day × hour mean) | 277.16 | 497.82 | 0.9361 |
| Random Forest Regression (100 trees) | 284.22 | 492.46 | 0.9375 |
| Feed-Forward Neural Network | 305.17 | 503.71 | 0.9346 |
| Extra Trees Regression | 308.59 | 540.82 | 0.9246 |
| Gradient Boosting Regression | 359.58 | 561.20 | 0.9189 |
| Decision Tree Regression | 379.87 | 672.17 | 0.8836 |
| KNN Regression | 532.84 | 812.94 | 0.8297 |
| Ridge Regression | 1325.11 | 1528.90 | 0.3977 |
| Linear Regression | 1325.13 | 1528.92 | 0.3977 |

**Interpretation:** Random Forest achieved the best RMSE and R² among the evaluated models, while the day-of-week-and-hour baseline achieved the lowest MAE.

The deployed application uses a smaller 20-tree Random Forest model to reduce the model file size.

| Deployed model metric | Value |
|---|---:|
| MAE | 289.89 |
| RMSE | 502.71 |
| R² | 0.9349 |

## Web Application

The Streamlit application allows users to enter weather and time-related information and obtain an estimated traffic volume prediction.

**Live application:** [Open the Traffic Volume Prediction App](https://sandhyabaliga-traffic-volume-prediction-app-vojpz0.streamlit.app/)

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Git and GitHub

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/sandhyabaliga/traffic-volume-prediction.git
cd traffic-volume-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Limitations

- Predictions are estimates based on historical data and the trained model.
- The application does not use live traffic feeds.
- Unusual events, road closures, accidents, and changing traffic patterns may affect real-world accuracy.
- The deployed model uses fewer trees than the original 100-tree Random Forest experiment.

## Future Enhancements

- Integrate real-time traffic data.
- Explore additional traffic and weather features.
- Evaluate time-series forecasting methods.
- Improve prediction performance during rush hours.
- Monitor performance as traffic patterns change.

## Author

**Sandhya Baliga**

MCA Student — Manipal Institute of Technology, Manipal

## Acknowledgements

Thanks to the UCI Machine Learning Repository for providing the Metro Interstate Traffic Volume Dataset.
