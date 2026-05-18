import logging
import os
from scripts.data_pipeline import preprocess_pipeline
from scripts.model_engine import prepare_prophet_matrix, extract_holidays, train_baseline_prophet, generate_forecast

# Configure logging for main execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DATA_PATH = os.getenv('DATA_PATH', 'data/daily_records.csv')
DATE_COL = 'date'
TARGET_COL = 'complaints'
FORECAST_HORIZON = 90
OUTPUT_PATH = 'data/production_forecast.csv'

def main():
    logger.info("--- Starting Operational Forecasting Pipeline ---")
    
    # 1. Data Engineering
    df_engineered = preprocess_pipeline(DATA_PATH, DATE_COL, TARGET_COL)
    
    # 2. Structural Matrix Formatting
    prophet_train_df = prepare_prophet_matrix(df_engineered, DATE_COL, TARGET_COL)
    prophet_holidays = extract_holidays(df_engineered, DATE_COL)
    
    # 3. Model Training
    model = train_baseline_prophet(prophet_train_df, prophet_holidays)
    
    # 4. Inference Generation
    forecast_output = generate_forecast(model, FORECAST_HORIZON)
    
    # 5. Output Serialization
    forecast_output.tail(FORECAST_HORIZON).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Pipeline executed successfully. Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()