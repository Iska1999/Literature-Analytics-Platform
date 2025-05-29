from orbit.models import LGT
import numpy as np

def fit_bsts_model(
    train_df,
    response_col,
    date_col,
    regressor_cols=None, # List of regressor column names
    model_type='LGT',    # 'LGT', 'DLT', 'KTR' etc.
    seasonality_period=None, # e.g., 12 for monthly data with yearly seasonality
    num_warmup_mcmc=500,     # For MCMC estimator
    n_bootstrap_draws_mcmc=1000, # For MCMC estimator
    verbose_fit=False
    ):
    """
    Defines, initializes, and fits an a bsts model using the Orbit module.
    """
    corr_matrix = train_df[regressor_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    filtered_cols = [col for col in regressor_cols if col not in to_drop]
    
    print(f"\nDefining and fitting Orbit {model_type} model...")
    if model_type == 'LGT':
        model = LGT(
            response_col=response_col,
            date_col=date_col,
            regressor_col=filtered_cols if filtered_cols else None,
            seasonality=seasonality_period if seasonality_period else None,
            estimator='stan-mcmc', # 'pyro-svi' for faster Variational Inference
            num_warmup=num_warmup_mcmc,
            n_bootstrap_draws=n_bootstrap_draws_mcmc, # Renamed from num_sample to n_bootstrap_draws in some versions
            verbose=verbose_fit
        )
    else:
        print(f"Error: Model type '{model_type}' not supported in this function.")
        return None

    try:
        model.fit(df=train_df)
        print(f"BSTS {model_type} model fitting complete.")
        return model
    except Exception as e:
        err_message = f"Error during BSTS model fitting. Check console for more details."
        print(err_message)
        import traceback
        traceback.print_exc()
        return err_message