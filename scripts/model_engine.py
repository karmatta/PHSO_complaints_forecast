import logging
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

def prepare_prophet_matrix(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Renames columns to match Prophet's strict internal schema."""
    return df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})

def extract_holidays(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Creates the holiday dataframe required by Prophet's initialization."""
    holidays_df = df[df['bank_holiday_flag'] == 1][[date_col]].rename(columns={date_col: 'ds'})
    holidays_df['holiday'] = 'uk_bank_holiday'
    return holidays_df

def train_baseline_prophet(df_train: pd.DataFrame, df_holidays: pd.DataFrame) -> Prophet:
    """Initializes and trains the production univariate Prophet baseline."""
    logger.info("Initializing Prophet Baseline Engine...")
    model = Prophet(
        holidays=df_holidays, 
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False
    )
    
    model.fit(df_train)
    logger.info("Prophet model fitting complete.")
    return model

def generate_forecast(model: Prophet, horizon: int) -> pd.DataFrame:
    """Generates the future grid and computes the forward forecast."""
    future_grid = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future_grid)
    
    # Clip non-physical negative predictions to zero
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    logger.info(f"Successfully generated {horizon}-day forecast.")
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]