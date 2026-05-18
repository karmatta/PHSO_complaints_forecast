import logging
import pandas as pd
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str, date_col: str) -> pd.DataFrame:
    """Loads raw CSV data and parses dates."""
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        logger.info(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def build_chronological_spine(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Creates a continuous daily calendar spine to expose implicit missing dates."""
    min_date, max_date = df[date_col].min(), df[date_col].max()
    spine = pd.date_range(start=min_date, end=max_date, freq='D')
    return pd.DataFrame({date_col: spine})

def clean_and_impute(df_raw: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Merges raw data onto the spine and imputes missing target values."""
    df_spine = build_chronological_spine(df_raw, date_col)
    df_clean = pd.merge(df_spine, df_raw, on=date_col, how='left')
    
    # Linear interpolation followed by boundary edge-case backfill/forwardfill
    df_clean[target_col] = df_clean[target_col].interpolate(method='linear').bfill().ffill()
    logger.info("Successfully aligned timeline and imputed missing targets.")
    return df_clean

def attach_holidays(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Generates deterministic UK Bank Holiday flags."""
    unique_years = list(df[date_col].dt.year.unique())
    uk_holidays = holidays.UnitedKingdom(subdiv='England', years=unique_years)
    
    df['bank_holiday_flag'] = df[date_col].isin(uk_holidays).astype(int)
    return df

def preprocess_pipeline(filepath: str, date_col: str, target_col: str) -> pd.DataFrame:
    """Executes the end-to-end data preparation pipeline."""
    df_raw = load_data(filepath, date_col)
    df_clean = clean_and_impute(df_raw, date_col, target_col)
    df_final = attach_holidays(df_clean, date_col)
    return df_final