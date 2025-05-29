import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from utils.useful_functions import inverse_transform

# Stationarity function
def check_stationarity_and_transform(df_input, field_features, sector_return_col, cols_to_diff=[]):
    df_analysis = df_input.copy()
    diff_counter = 0
    if df_analysis.isnull().any().any():
        print("Warning: Input DataFrame contains NaNs. Dropping them.")
        df_analysis.dropna(inplace=True)

    if len(df_analysis) < 30:
        print(f"Error: Not enough data points ({len(df_analysis)}) for reliable VAR analysis.")
        return None
    
    df_stationary = df_analysis.copy()
    cols_to_drop = []

    for col in df_stationary.columns:
        original_series = df_stationary[col].copy().dropna()
        adf_result = adfuller(original_series)
        print(f"ADF p-value for {col}: {adf_result[1]:.4f}")

        if adf_result[1] > 0.05 or col in cols_to_diff:
            df_stationary[col] = df_stationary[col].diff()
            adf_result_diff = adfuller(df_stationary[col].dropna())
            print(f"  - Differenced. New ADF p-value for {col}: {adf_result_diff[1]:.4f}")
            if col == sector_return_col:
                diff_counter = 1
            if adf_result_diff[1] > 0.05 and len(df_stationary[col].dropna()) > 10:
                print(f"  - Still non-stationary, trying second difference for {col}")
                df_stationary[col] = df_stationary[col].diff()
                adf_result_diff2 = adfuller(df_stationary[col].dropna())
                print(f"    - Second diff. New ADF p-value for {col}: {adf_result_diff2[1]:.4f}")
                if col == sector_return_col:
                    diff_counter = 2
                if adf_result_diff2[1] > 0.05:
                    print(f"    - {col} still non-stationary after second differencing. Dropping.")
                    cols_to_drop.append(col)
            elif adf_result_diff[1] > 0.05:
                print(f"  - {col} still non-stationary after first differencing. Dropping.")
                cols_to_drop.append(col)
        else:
            print(f"  - {col} is stationary.")

    if cols_to_drop:
        print(f"\nDropping non-stationary columns: {cols_to_drop}")
        df_stationary.drop(columns=cols_to_drop, inplace=True)

    return df_stationary.dropna(),diff_counter

def invert_first_difference(forecast_diff, last_actual_value):
    """Reconstructs the original series from the differenced forecast"""
    forecast_diff = np.array(forecast_diff).flatten()
    inverted = np.r_[last_actual_value, forecast_diff].cumsum()
    return inverted  # exclude the starting value itself


def preprocess_data_for_var(
    df_input: pd.DataFrame,
    field_features,
    target_col: str, # Specify your main target variable for VIF context (optional but good)
    stationarity_sig_level: float = 0.05,
    near_zero_var_threshold: float = 1e-6, # Standard deviation threshold
    correlation_threshold: float = 0.85,   # For dropping highly correlated features
    vif_threshold: float = 10.0,           # For VIF check
    plot_stationarity_transformations: bool = False,
    plot_correlation_heatmap: bool = False,
    output_plot_prefix: str = "preprocessing"
):
    """
    Preprocesses a DataFrame for VAR modeling:
    1. Handles initial NaNs.
    2. Checks for and attempts to correct non-stationarity.
    3. Removes near-zero variance columns.
    4. Checks for multicollinearity (correlation and VIF) and suggests/drops features.

    Args:
        df_input (pd.DataFrame): The input DataFrame with time series.
        target_col (str): Name of the primary target variable (e.g. sector return).
                          Used for context in VIF if calculated iteratively.
        stationarity_sig_level (float): Significance level for ADF test.
        near_zero_var_threshold (float): Std dev below which a column is considered near-zero variance.
        correlation_threshold (float): Absolute correlation above which one of a pair is dropped.
        vif_threshold (float): VIF score above which a feature is considered highly collinear.
        plot_stationarity_transformations (bool): Whether to plot original vs. transformed series.
        plot_correlation_heatmap (bool): Whether to plot the correlation heatmap.
        output_plot_prefix (str): Prefix for saving plot filenames.

    Returns:
        tuple: (pd.DataFrame or None, dict)
               - Processed DataFrame ready for VAR (or None if critical errors).
               - Report dictionary detailing actions taken and issues found.
    """
    report = {
        "initial_columns": df_input.columns.tolist(),
        "dropped_columns": [],
        "stationarity_transformations": {},
        "multicollinearity_removed": [],
        "vif_issues_found": [],
        "notes": []
    }
    df_processed = df_input.copy()

    if df_processed.empty:
        report["notes"].append("Input DataFrame is empty.")
        return None, report

    # --- 1. Handle Initial NaNs (e.g., from prior pct_change or rolling functions) ---
    initial_len = len(df_processed)
    df_processed.dropna(axis=0, how='any', inplace=True) # Drop rows with any NaNs
    if len(df_processed) < initial_len:
        report["notes"].append(f"Dropped {initial_len - len(df_processed)} rows due to initial NaNs.")
    
    if len(df_processed) < 20: # Arbitrary minimum for further processing
        report["notes"].append("Not enough data after initial NaN drop for robust preprocessing.")
        return None, report


    # --- 2. Stationarity ---
    print("\n--- Checking Stationarity ---")
    df_processed,diff_counter = check_stationarity_and_transform(df_processed,field_features,target_col,cols_to_diff=[])

    if df_processed.empty or len(df_processed) < 20 :
        report["notes"].append("Not enough data after stationarity transformations and NaN drop.")
        return None, report
    
    report["columns_after_stationarity"] = df_processed.columns.tolist()

    # --- 3. Near-Zero Variance ---
    print("\n--- Checking for Near-Zero Variance Columns ---")
    cols_to_drop_nzv = []
    for col in df_processed.columns:
        if df_processed[col].std() < near_zero_var_threshold:
            cols_to_drop_nzv.append(col)
            report["dropped_columns"].append(f"{col} (near-zero variance: std={df_processed[col].std():.2e})")
    
    if cols_to_drop_nzv:
        df_processed.drop(columns=cols_to_drop_nzv, inplace=True)
        print(f"Dropped near-zero variance columns: {cols_to_drop_nzv}")
        if df_processed.empty:
            report["notes"].append("All columns dropped due to near-zero variance.")
            return None, report
    else:
        print("No near-zero variance columns found.")
    report["columns_after_nzv_check"] = df_processed.columns.tolist()


    # --- 4. Multicollinearity ---
    if len(df_processed.columns) < 2:
        print("Less than 2 columns remaining, skipping multicollinearity check.")
        report["final_columns"] = df_processed.columns.tolist()
        return df_processed, report

    print("\n--- Checking for Multicollinearity ---")
    # a. High Correlation Check
    corr_matrix = df_processed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Get upper triangle
    
    cols_to_drop_corr = set()
    for column in upper.columns:
        highly_correlated_with_column = upper[upper[column] > correlation_threshold].index
        for correlated_feature in highly_correlated_with_column:
            # Basic strategy: keep the one that's potentially the target, or just the first one encountered
            # A more sophisticated strategy might involve domain knowledge or VIF scores.
            if column == target_col and correlated_feature != target_col: # Keep target
                cols_to_drop_corr.add(correlated_feature)
                report["multicollinearity_removed"].append(f"{correlated_feature} (highly correlated with {column}, r={corr_matrix.loc[column, correlated_feature]:.2f})")
            elif correlated_feature == target_col and column != target_col: # Keep target
                 cols_to_drop_corr.add(column)
                 report["multicollinearity_removed"].append(f"{column} (highly correlated with {correlated_feature}, r={corr_matrix.loc[column, correlated_feature]:.2f})")
            elif column not in cols_to_drop_corr and correlated_feature not in cols_to_drop_corr : # Arbitrarily drop the second one in the pair
                cols_to_drop_corr.add(correlated_feature) # Drop the one in the row index
                report["multicollinearity_removed"].append(f"{correlated_feature} (highly correlated with {column}, r={corr_matrix.loc[column, correlated_feature]:.2f})")

    if cols_to_drop_corr:
        # Ensure target_col is not accidentally added to cols_to_drop_corr if it's part of a high-corr pair
        cols_to_drop_corr.discard(target_col)
        if cols_to_drop_corr: # Check again after possibly removing target_col
            df_processed.drop(columns=list(cols_to_drop_corr), inplace=True)
            print(f"Dropped due to high correlation: {list(cols_to_drop_corr)}")
            if df_processed.empty or len(df_processed.columns) < 2:
                report["notes"].append("Not enough columns remaining after correlation drop.")
                report["final_columns"] = df_processed.columns.tolist()
                return df_processed if not df_processed.empty else None, report
    else:
        print("No features dropped due to pairwise correlation above threshold.")

    report["final_columns"] = df_processed.columns.tolist()
    report["final_shape"] = df_processed.shape
    print("\n--- Preprocessing Complete ---")
    print(f"Final columns for VAR: {report['final_columns']}")
    print(f"Final DataFrame shape: {report['final_shape']}")
    
    return df_processed, report,diff_counter

def fit_var_model(df_stationary,feasible_max_lags=8):
    err_message = 0 
    # 1. Select VAR Order & Fit Model  <--- LAG SELECTION IS PART OF THIS BLOCK
    var_results = None
    best_lag = 0 # Will be determined
    try:
        model_instance = VAR(df_stationary) # df_stationary contains ALL variables for this VAR
        
        # ... (calculation of feasible_max_lags) ...

        # THIS IS WHERE LAG ORDER IS ASSESSED AND SELECTED:
        selected_orders = model_instance.select_order(maxlags=feasible_max_lags)
        #print(f"\nVAR Lag Order Selection Summary for {pairing_label}:")
        print(selected_orders.summary())
        
        best_lag = selected_orders.aic # Or selected_orders.bic, etc.
        # ... (logic to handle if best_lag is a dict or 0) ...

        #print(f"Selected VAR lag order (AIC) for {pairing_label}: {best_lag}")
        
        # THE FINAL MODEL IS THEN FIT WITH THIS best_lag:
        var_results = model_instance.fit(best_lag) # <-- Fitting with the chosen lag
        #print(f"\nVAR Model Summary for {pairing_label}:\n{var_results.summary()}")
        if var_results.k_ar == 0: # Check if number of lags is zero
            err_message = "Error: VAR model has 0 lags."
            print(err_message)
            return err_message,var_results
        elif not var_results.is_stable():
            err_message = "Error: VAR model is not stable. IRFs may be unreliable or fail."
            print("Warning: VAR model is not stable. IRFs may be unreliable or fail.")
            return err_message,var_results
        else:
            return "",var_results
    except Exception as e:
        err_message = "Error during VAR model fitting or lag selection."
        print(err_message)
        return err_message, var_results # Return partial if fitting failed after selection

def var_forecast_plot(results, df,train_df, test_df, target_col, scalers,diff_counter,forecast_steps=4,test_start_month=None):
    """
    Performs forecasting, and plots actual vs predicted.

    Parameters:
    - df: DataFrame with a datetime index (monthly) and multiple numeric columns.
    - target_col: str, column to forecast (e.g., 'price_change')
    - forecast_steps: int, how many months to forecast ahead
    - test_start_month: str in format "YYYY-MM", the month to start forecasting (optional)

    Returns:
    - fig: figure 
    - metrics: forecast metrics
    """
    #df = pd.concat([train_df, test_df])
    # Forecast
    lag_order = results.k_ar
    forecast_input_df_ordered = train_df[results.model.endog_names] # Select and reorder columns
    forecast_input = forecast_input_df_ordered.values[-lag_order:]
    forecast,lower,upper = results.forecast_interval(forecast_input, steps=forecast_steps, alpha=0.05)
    
    # Construct DataFrames
    forecast_df = pd.DataFrame(forecast, columns=train_df.columns, index=test_df.index[:forecast_steps])
    lower_df = pd.DataFrame(lower, columns=train_df.columns, index=test_df.index[:forecast_steps])
    upper_df = pd.DataFrame(upper, columns=train_df.columns, index=test_df.index[:forecast_steps])
    
    # Unscale if scalers are provided
    if scalers:
        forecast_df[target_col] = scalers[target_col].inverse_transform(forecast_df[[target_col]])
        lower_df[target_col] = scalers[target_col].inverse_transform(lower_df[[target_col]])
        upper_df[target_col] = scalers[target_col].inverse_transform(upper_df[[target_col]])
        test_df[target_col] = scalers[target_col].inverse_transform(test_df[[target_col]])
    
    #Need a check on whether i should or not
    if (diff_counter!=0):
        last_actual_value = df[target_col].iloc[-5]
        undiff_forecast_lower = invert_first_difference(forecast_diff=lower_df[target_col], last_actual_value=last_actual_value)
        undiff_forecast_upper = invert_first_difference(forecast_diff=upper_df[target_col], last_actual_value=last_actual_value)
        undiff_forecast_values = invert_first_difference(forecast_diff=forecast_df[target_col], last_actual_value=last_actual_value)
        lower_df[target_col] = undiff_forecast_lower[1:]
        forecast_df[target_col] = undiff_forecast_values[1:]
        upper_df[target_col] = undiff_forecast_upper[1:]
    
    # Plot
    # Plot actual vs forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[target_col], label="Actual",color='blue',marker='o')
    ax.plot(forecast_df[target_col], label="Forecast", marker='o', linestyle='--', color='orange')
    ax.axvline(x=test_df.index[0], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title(f"Forecast vs Actual: {target_col} (Unscaled)")
    ax.fill_between(forecast_df.index, lower_df[target_col], upper_df[target_col], color='orange', alpha=0.3, label="95% CI")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.legend()
    plt.xticks(rotation=90) 
    fig.tight_layout()
    
    #actual = test_df[target_col]#.loc[forecast_df.index]
    actual = df[target_col].iloc[-4:]
    predicted = forecast_df[target_col]

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)*100

    metrics = metrics = pd.DataFrame([{
    "MAE": round(mae, 3),
    "MSE": round(mse, 3),
    "RMSE": round(rmse, 3),
    'MAPE (%)': round(mape, 3)
    }])

    return fig,metrics


