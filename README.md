# AI-Powered Sales Forecasting

This project implements a dual-model time series forecasting pipeline to predict future sales revenue using historical retail transaction data. It combines classical statistical modeling (ARIMA) with deep learning (LSTM) to generate robust and interpretable forecasts. A Streamlit dashboard provides an interactive interface to visualize trends and predictions.

---

## Project Objective

To enable business teams and data scientists to:
- Forecast monthly revenue using both ARIMA and LSTM models
- Compare model performance using visual and quantitative metrics
- Provide actionable business forecasts through an interactive dashboard

---

## Features

- Time series aggregation and preprocessing
- LSTM-based forecasting (sequence modeling)
- ARIMA-based univariate prediction
- Evaluation with MAE and RMSE
- Interactive dashboard for trend exploration
- CSV export of forecast results

---

## Conceptual Study

For a deeper dive into the model designs and forecasting logic, read the [Conceptual Study PDF](https://github.com/Pre123140/Sales-Forecasting/blob/main/AI_POWERED_SALES_FORECASTING.pdf).

Covers:
- ARIMA hyperparameter tuning and seasonality
- LSTM input preparation and architecture
- Model comparison logic and interpretation
- Forecasting challenges in real-world applications

---

## Tech Stack

- pandas – Data loading and manipulation
- numpy – Time series aggregation and transformation
- scikit-learn – MinMaxScaler and evaluation metrics
- tensorflow.keras – Deep learning model (LSTM)
- pmdarima – Auto ARIMA for optimal statistical forecasting
- statsmodels – Time series modeling backend
- matplotlib, seaborn – Static data visualization
- plotly – Interactive charts
- streamlit – Interactive dashboard interface

---

## Folder Structure

```
SALES_FORECASTING_ADVANCED/
├── cleaned_Superstore.csv
├── forecast_results.csv
├── processed_superstore.csv
├── sales_forecasting.py
├── Superstore_Feature_Engineered.csv
├── Superstore.csv
├── requirements.txt

```

Note: All outputs are displayed through the dashboard. No images or graphs are stored in the repository.

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Pre123140/SALES_FORECASTING_PROJECT.git
cd SALES_FORECASTING_ADVANCED
```

### 2. (Optional) Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run sales_forecasting.py
```

---

## Model Performance Summary

| Model | MAE       | RMSE      |
|--------|------------|-----------|
| LSTM   | 16,163.11  | 18,025.34 |
| ARIMA  | Similar, interpretable trendlines |

- LSTM effectively captured recent and nonlinear trends
- ARIMA captured seasonal and long-term behavior
- Both predictions are visualized side-by-side in the dashboard

---

## Key Deliverables

- Time series feature engineered datasets
- Forecasted revenue using both LSTM and ARIMA
- MAE/RMSE evaluation of both models
- Streamlit-based forecasting dashboard
- Downloadable CSV of predictions

---

## Future Enhancements

- Add Facebook Prophet as a third forecasting option
- Enable regional or subcategory-based forecasts
- PDF/Excel report generation for sharing insights
- Add deployment-ready UI and container support
- Use hybrid/ensemble models for improved accuracy

---

## License

This project is open for educational use only. For commercial deployment, contact the author.

---

## Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).
