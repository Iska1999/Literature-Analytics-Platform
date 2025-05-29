#Function to count unique authors
def count_unique_authors(series):
    all_authors = [author for sublist in series.dropna() for author in sublist]
    return len(set(all_authors))

#Counts # publications
def count_publications(series):
    return series.count()

#Counts number of subcategories unique from the paper's main field
def compute_diversity(row):
    import ast

    # Parse stringified list if needed
    if isinstance(row['categories'], str):
        try:
            categories = ast.literal_eval(row['categories'])
        except Exception:
            categories = []
    else:
        categories = row['categories']

    field = row['field'].lower().strip()

    # Count categories where field is NOT a substring
    diversity_count = sum(
        1 for cat in categories if field not in cat.lower()
    )

    return diversity_count

def compute_monthly_growth_trends(df, value_cols, group_col="field", window=12):
    """
    Computes:
    1. Monthly percent change per group
    2. Rolling average (12-month) per group
    3. Rolling growth rate (% change from 12 months ago) per group
    
    Returns:
    - df_pct: with monthly percent change
    - df_full: with all computed features
    """
    df_sorted = df.sort_values(by=["month", group_col]).copy()

    # Step 1: Monthly percent change
    for col in value_cols:
        change_col = f"{col}_pct_change"
        df_sorted[change_col] = (
            df_sorted.groupby(group_col)[col]
            .pct_change()
            .round(4)
        )

    df_pct = df_sorted.copy()

    # Step 2: Rolling statistics
    for col in value_cols:
        # Rolling average within group
        roll_avg_col = f"{col}_rolling_avg"
        df_sorted[roll_avg_col] = (
            df_sorted.groupby(group_col)[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            .round(4)
        )

        # Rolling growth rate within group (lagged % change)
        roll_growth_col = f"{col}_growth_rate"
        df_sorted[roll_growth_col] = (
            df_sorted.groupby(group_col)[col]
            .transform(lambda x: x.pct_change(periods=window))
            .round(4)
        )

    # Step 3: Drop rows with NaNs in any new columns
    return df_pct.dropna().reset_index(drop=True), df_sorted.dropna().reset_index(drop=True)