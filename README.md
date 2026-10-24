# Traffic Volume Prediction Using Machine Learning

## Project Overview

Traffic Volume Prediction is a machine learning project that predicts traffic volume using historical traffic data, weather conditions, and time-based features.

The project uses the **Metro Interstate Traffic Volume Dataset** and explores different machine learning regression models to predict traffic volume. It includes data preprocessing, exploratory data analysis, feature engineering, model training, model evaluation, and a Streamlit web application for interactive predictions.

## Objectives

- Analyze historical traffic volume patterns.
- Study the relationship between traffic volume, weather, and time.
- Perform data preprocessing and feature engineering.
- Train and compare multiple machine learning regression models.
- Evaluate model performance using MAE, RMSE, and R² score.
- Deploy a web application for traffic volume prediction.

## Dataset

**Dataset:** Metro Interstate Traffic Volume

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

The original dataset contains 48,204 records and 9 columns.

Important attributes include:

- `date_time` — date and time of observation
- `traffic_volume` — recorded traffic volume
- `temp` — temperature
- `rain_1h` — rainfall during one hour
- `snow_1h` — snowfall during one hour
- `clouds_all` — cloud coverage
- `weather_main` — general weather condition
- `weather_description` — detailed weather condition
- `holiday` — holiday information

## Methodology

### 1. Data Preprocessing

- Removed 17 exact duplicate records.
- Aggregated records with identical timestamps.
- Sorted observations chronologically.
- Split the data into 80% training data and 20% testing data.
- Used a preprocessing pipeline for missing-value imputation, numerical feature scaling, and categorical feature encoding.

After timestamp aggregation, the dataset contained **40,575 records**.

### 2. Feature Engineering

The following features were used for prediction:

- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- General weather condition
- Hour of the day
- Day of the week
- Month
- Weekend indicator
- Rush-hour indicator
- Holiday indicator

### 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand traffic patterns across different hours, days of the week, months, and weather conditions.

Key observations included:

- Average traffic volume was highest around 4 PM.
- Average traffic volume was lowest around 3 AM.
- Friday had the highest average traffic volume among the days of the week analyzed.
- Sunday had the lowest average traffic volume among the days of the week analyzed.

### 4. Machine Learning Models

The following models were evaluated:

1. Baseline Prediction (day-of-week and hour mean)
2. Random Forest Regression
3. Feed-Forward Neural Network (MLP Regressor)
4. Extra Trees Regression
5. Gradient Boosting Regression
6. Decision Tree Regression
7. K-Nearest Neighbors (KNN) Regression
8. Ridge Regression
9. Linear Regression

## Model Evaluation

The models were evaluated using three regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted traffic volume. Lower is better.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily. Lower is better.
- **R² Score:** Measures how well the model explains variation in the target. Higher is better.

### Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Baseline (day × hour mean) | 277.16 | 497.82 | 0.9361 |
| Random Forest Regression | 284.22 | 492.46 | 0.9375 |
| Feed-Forward Neural Network | 305.17 | 503.71 | 0.9346 |
| Extra Trees Regression | 308.59 | 540.82 | 0.9246 |
| Gradient Boosting Regression | 359.58 | 561.20 | 0.9189 |
| Decision Tree Regression | 379.87 | 672.17 | 0.8836 |
| KNN Regression | 532.84 | 812.94 | 0.8297 |
| Ridge Regression | 1325.11 | 1528.90 | 0.3977 |
| Linear Regression | 1325.13 | 1528.92 | 0.3977 |

### Results and Interpretation

- The **Baseline model** achieved the lowest MAE of 277.16.
- **Random Forest Regression** achieved the lowest RMSE of 492.46 and the highest R² score of 0.9375 among the evaluated models.
- The Feed-Forward Neural Network also performed well, achieving an R² score of 0.9346.
- Ridge Regression and Linear Regression performed less effectively than the tree-based models in these experiments.

Random Forest Regression was selected for the deployed application.

## Deployed Model

To reduce the model file size for deployment, the application uses a smaller Random Forest model with 20 trees.

| Metric | Value |
|---|---:|
| MAE | 289.89 |
| RMSE | 502.71 |
| R² Score | 0.9349 |

These metrics correspond to the deployed 20-tree model, while the main comparison table reports the original model experiments.

## Web Application

The project includes an interactive web application built with Streamlit.

Users can enter time-related and weather-related information to obtain an estimated traffic volume prediction.

**Live Demo:** [Traffic Volume Prediction App](https://sandhyabaliga-traffic-volume-prediction-app-vojpz0.streamlit.app/)

The application provides predictions from the trained model. It does not use live traffic feeds.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Version Control:** Git and GitHub

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/sandhyabaliga/traffic-volume-prediction.git
cd traffic-volume-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

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

The application will open in your browser.

## Limitations

- Predictions are based on historical data and learned patterns.
- The application does not use real-time traffic information.
- Accidents, road closures, special events, and sudden weather changes may affect actual traffic conditions.
- Model performance may change when traffic patterns differ from the training data.

## Future Enhancements

- Integrate real-time traffic data.
- Explore additional traffic and weather features.
- Evaluate time-series forecasting techniques.
- Improve prediction accuracy during rush hours.
- Monitor model performance using newer traffic data.

## Author

**Sandhya Baliga**  
MCA Student  
Manipal Institute of Technology, Manipal

## Acknowledgements

The Metro Interstate Traffic Volume Dataset is provided by the UCI Machine Learning Repository.
