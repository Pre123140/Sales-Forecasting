# 📊 AI-Powered Sales Forecasting

**Predict future revenue trends using deep learning (LSTM) and statistical forecasting (ARIMA)**

---

## 📌 Overview

This project builds a robust sales forecasting pipeline using two powerful time-series approaches:
- **LSTM (Long Short-Term Memory)**: A deep learning model capable of learning complex sequential patterns
- **ARIMA/SARIMA**: A classical statistical model for structured univariate forecasting

The goal is to forecast monthly sales revenue using historical retail transaction data and visualize predictions through an interactive **Streamlit dashboard**.

---

## 🎯 Project Objective

- Build a time-series forecasting model for monthly sales
- Compare statistical and deep learning approaches (ARIMA vs LSTM)
- Present forecasts in a business-friendly dashboard with downloadable results
- Deliver explainable and actionable outputs for business decision-making

---

## 🧠 Technologies & Libraries

| Tool | Purpose |
|------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn` | Exploratory data visualization |
| `MinMaxScaler` | Normalize values for LSTM |
| `tensorflow.keras` | Deep learning (LSTM) model |
| `pmdarima`, `statsmodels` | Time series forecasting (ARIMA) |
| `scikit-learn` | Evaluation metrics |
| `streamlit`, `plotly` | Dashboard and visual reporting |

---

## 🗂️ Project Structure
📁 AI-Sales-Forecasting-Project/
├── salesforecasting.py               # ✅ Main implementation script (LSTM + ARIMA + Streamlit)
│
├── data/                             # 📦 Raw and processed datasets
│   ├── Superstore.csv                # Original dataset
│   ├── cleaned_Superstore.csv        # Cleaned & monthly aggregated sales data
│   ├── processed_superstore.csv      # Dashboard-ready sales data
│   ├── forecast_results.csv          # 12-month ARIMA forecast results
│
├── output/                           # 📊 Visualizations & dashboard screenshots
│   ├── Figure_1.png                  # Scaled monthly sales trend
│   ├── Figure_2.png                  # Actual vs. predicted sales (LSTM)
│   ├── Figure_3.png                  # ARIMA forecast with confidence intervals
│   ├── dashboard_screenshot_1.png   # Streamlit metrics view
│   ├── dashboard_screenshot_2.png   # Streamlit forecast visualization
│
├── README.md                         # 📘 Project overview and instructions (GitHub read

---

## 📈 Data Processing & Modeling Flow

1. **Data Aggregation**
   - Daily retail sales data resampled to monthly
   - Cleaned, sorted, and indexed by date

2. **LSTM Forecasting**
   - Scaled using `MinMaxScaler`
   - Created 12-month rolling sequences
   - Trained LSTM model with two layers, dropout regularization

3. **ARIMA Forecasting**
   - Best parameters selected via `auto_arima`
   - Forecasted 12 months ahead with confidence intervals

4. **Evaluation**
   - Metrics: MAE, RMSE
   - Visual plots: actual vs predicted (LSTM), forecast chart (ARIMA)

5. **Dashboard**
   - Built with Streamlit to showcase trends, forecasts, and allow CSV export

---

## 📊 Results

| Metric | Value |
|--------|--------|
| MAE    | 16,163.11 |
| RMSE   | 18,025.34 |

- LSTM captured short-term fluctuations effectively
- ARIMA showed interpretable seasonal patterns
- Both models were plotted for comparison
- Business-ready dashboard built for analysis

---

## 🚀 Try It Yourself

1. Install required packages:

```bash
pip install -r requirements.txt


2. Run the Streamlit dashboard:
streamlit run salesforecasting.py

Deliverables
✅ Forecasted datasets (CSV)
✅ Interactive Streamlit app
✅ Evaluation metrics
✅ Visual comparisons of both models
✅ PDF reports and conceptual explanation (separate)

📘 Learn More
Explore the full Conceptual Study to understand:
Why LSTM and ARIMA were chosen
Time-series modeling principles
Real-world forecasting applications

📄 License & Use
This project is for educational and illustrative purposes only.
Commercial use or adaptation is prohibited without permission.


---

Let me know if you'd like a **separate `requirements.txt`**, PDF export of the README, or if you'd like the dashboard hosted for preview.

