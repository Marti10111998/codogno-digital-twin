import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Historical Data ---
historical_file = "/Users/martinaivanova/Desktop/Merged2.csv"
historical_df = pd.read_csv(historical_file)

# --- Load Forecast Data ---
forecast_file = "/Users/martinaivanova/Desktop/30dayNO2.csv"
forecast_df = pd.read_csv(forecast_file)

# --- Convert DateTime to datetime format ---
historical_df["DateTime"] = pd.to_datetime(historical_df["DateTime"], format="%m/%d/%y %H:%M")
forecast_df["DateTime"] = pd.to_datetime(forecast_df["DateTime"], format="%m/%d/%y %H:%M")

# --- Extract Time Features ---
for df in [historical_df, forecast_df]:
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Year"] = df["DateTime"].dt.year
    df["Weekday"] = df["DateTime"].dt.weekday

# --- Fix column name issues in forecast data ---
forecast_df.columns = forecast_df.columns.str.strip()

# --- Ask user to select target variable ---
print("Available columns for prediction:")
print([col for col in historical_df.columns if col != "DateTime"])
target_variable = input("Enter the target variable (y) to predict from the available columns: ")

# --- Define feature set dynamically ---
features = [col for col in historical_df.columns if col not in ["DateTime", target_variable]]
X = historical_df[features]
y = historical_df[target_variable]

# --- Define train/test split values ---
train_test_splits = [0.5, 0.6, 0.7, 0.8, 0.9]

# --- Define models ---
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
}

# --- Loop over different models and generate files ---
for name, model in models.items():
    all_results = []
    forecast_results = forecast_df.copy()
    
    for train_size in train_test_splits:
        test_size = 1 - train_size
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        all_results.append([train_size * 100, test_size * 100, mae, mse, r2])
        
        # Forecast Target Variable
        forecast_results[f"{target_variable}_Forecast_{int(train_size * 100)}"] = model.predict(forecast_results[features])
    
    # Save performance results
    results_df = pd.DataFrame(all_results, columns=["Train %", "Test %", "MAE", "MSE", "R2_Score"])
    results_filename = f"model_performance_{name}.csv"
    results_df.to_csv(results_filename, index=False)
    
    # Save forecast results
    forecast_filename = f"{target_variable}_forecast_{name}.csv"
    forecast_results.to_csv(forecast_filename, index=False)
    
    print(f"✅ Saved: {results_filename} and {forecast_filename}")

print("✅ All forecasting complete! Check the generated CSV files.")
