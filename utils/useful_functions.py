from sklearn.preprocessing import MinMaxScaler # Optional: if you want to scale regressors manually

def scale_data(df, regressor_cols, scale_regressors=True):

    df_processed = df.copy()
    scalers = {} # To store scalers if used

    if scale_regressors:
        print("Scaling columns...")
        for col in regressor_cols:
            scaler = MinMaxScaler(feature_range=(0, 1)) # Scale to -1, 1 can sometimes be good for regressors
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
            scalers[col] = scaler

    return df_processed, scalers

def transform_data(df, regressor_cols, scalers):
    """
    Transforms test data using previously fitted scalers.
    
    Parameters:
    - df: Test DataFrame to be transformed
    - regressor_cols: List of columns to transform
    - scalers: Dictionary of fitted scalers from training
    
    Returns:
    - Transformed DataFrame
    """
    df_transformed = df.copy()

    for col in regressor_cols:
        if col in scalers:
            df_transformed[col] = scalers[col].transform(df_transformed[[col]])
        else:
            raise ValueError(f"No fitted scaler found for column: {col}")

    return df_transformed

def inverse_transform(forecast_df, scalers, target_cols):
    """
    Inversely transforms scaled forecast data using original training scalers.
    
    Parameters:
    - forecast_df: DataFrame with scaled forecasts
    - scalers: dict of fitted scalers
    - target_cols: list of columns to inverse transform
    
    Returns:
    - forecast_df with new columns '<col>_unscaled'
    """
    df = forecast_df.copy()
    for col in target_cols:
        if col in scalers:
            df[f"{col}"] = scalers[col].inverse_transform(df[[col]])
        else:
            if col == 'month':
                pass
            else:
                raise ValueError(f"Scaler not found for {col}")
    return df
