import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess dataset
file_path = "cleaned_Superstore.csv"
df = pd.read_csv(file_path, parse_dates=["Order Date"])
df.sort_values(by="Order Date", inplace=True)
df.set_index("Order Date", inplace=True)

# Aggregate monthly sales
df_monthly = df.resample("M")["Sales"].sum()

# Normalize sales data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_monthly.values.reshape(-1, 1))
df_scaled = pd.DataFrame(df_scaled, index=df_monthly.index, columns=["Sales"])

# Plot scaled sales data
plt.figure(figsize=(12, 6))
sns.lineplot(x=df_scaled.index, y=df_scaled["Sales"], label="Scaled Sales")
plt.title("Monthly Sales Data (Scaled)")
plt.xlabel("Date")
plt.ylabel("Sales (Scaled)")
plt.legend()
plt.show()

# Function to create sequences
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
time_steps = 12
X, y = create_sequences(df_scaled.values, time_steps)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(100, activation="relu", return_sequences=True, input_shape=(12, 1)),
    Dropout(0.2),
    LSTM(100, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Plot actual vs predicted sales
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Sales", marker="o", linestyle="-")
plt.plot(y_pred, label="Predicted Sales", marker="x", linestyle="--")
plt.xlabel("Time (Months)")
plt.ylabel("Sales")
plt.title("Actual vs. Predicted Sales (LSTM Model)")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Auto ARIMA model selection
auto_arima_model = pm.auto_arima(
    df_monthly, seasonal=True, m=12, trace=True, error_action="ignore", suppress_warnings=True, stepwise=True
)
best_order = auto_arima_model.order
best_seasonal_order = auto_arima_model.seasonal_order
print(f"Best Model: ARIMA{best_order} Seasonal{best_seasonal_order}")

# Fit SARIMA model
sarima_model = SARIMAX(df_monthly, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
results = sarima_model.fit()

# Forecast
forecast_steps = 12
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df_monthly.index[-1], periods=forecast_steps + 1, freq="M")[1:]
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
sns.lineplot(x=df_monthly.index, y=df_monthly, label="Actual Sales", color="blue")
sns.lineplot(x=forecast_index, y=forecast_values, label="Forecasted Sales", color="red")
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
plt.title("Sales Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# Streamlit Dashboard
st.title("Sales Forecasting Dashboard")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# Sales Over Time Plot
st.subheader("Sales Over Time")
fig = px.line(df_monthly, y="Sales", title="Historical Sales Trends")
st.plotly_chart(fig)

# Forecast Visualization
st.subheader("Sales Forecast")
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly, mode='lines', name='Historical Sales'))
fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecasted Sales', line=dict(dash='dot')))
st.plotly_chart(fig_forecast)

# Download Data
st.subheader("Download Processed Data")
st.download_button("Download CSV", df_monthly.to_csv(index=True), file_name="processed_superstore.csv", mime="text/csv")
