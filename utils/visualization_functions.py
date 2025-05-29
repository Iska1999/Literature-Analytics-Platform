import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import graphviz
import pandas as pd
from utils.useful_functions import scale_data
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
import numpy as np

#Visualization Functions
def plot_time_series(df, date_col, y_cols):
    """
    Plots time series to visualize data
    """
    fig = go.Figure()
    
    df,scalers= scale_data(df, y_cols)

    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode='lines+markers',
            name=col
        ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Scales Values",
        hovermode="x unified",
        legend_title="Metric",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

def display_irf_plots(
    var_results, # : VARResultsWrapper,
    response_variable: str,
    impulse_variables: list,
    irf_periods: int = 24,
    n_cols: int = 2,
    title_fontsize: int = 14,
    label_fontsize: int = 12,     
    tick_fontsize: int = 10,      
    legend_fontsize: int = 10    
):
    """
    Displays Impulse Response Function (IRF) plots in Streamlit with larger fonts.
    """
    if not var_results or not hasattr(var_results, 'irf'):
        st.warning("VAR results object is not available or invalid for IRF plotting.")
        return

    try:
        irf_object = var_results.irf(periods=irf_periods)
    except Exception as e:
        st.error(f"Error generating IRF object: {e}")
        return

    if not impulse_variables:
        st.info("No impulse variables selected to plot IRFs for.")
        return

    cols = st.columns(min(n_cols, len(impulse_variables)))
    plot_idx = 0

    for impulse_var in impulse_variables:
        if impulse_var not in var_results.model.endog_names or \
           response_variable not in var_results.model.endog_names:
            with cols[plot_idx % n_cols]:
                st.warning(f"Skipping IRF for impulse '{impulse_var}' or response '{response_variable}' due to name mismatch in model.")
            plot_idx +=1
            continue
        
        try:
            # Generate plot using statsmodels' IRF plot method
            # It returns a matplotlib Figure object. We'll get the axes to customize.
            fig = irf_object.plot(
                impulse=impulse_var,
                response=response_variable,
                stderr_type='mc',
                repl=100,
                signif=0.05
            )
            
            if fig and fig.axes:
                ax = fig.axes[0] # Get the primary axes object

                # Customize fonts
                ax.set_title(f"Shock from: {impulse_var}", fontsize=title_fontsize-2) # Sub-title
                fig.suptitle(f"IRF: {impulse_var} -> {response_variable}", fontsize=title_fontsize)
                
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
                
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                if ax.get_legend() is not None:
                    plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)

                plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust for suptitle

                with cols[plot_idx % n_cols]:
                    st.pyplot(fig)
                plt.close(fig) 
            else:
                with cols[plot_idx % n_cols]:
                    st.warning(f"Could not generate IRF plot for impulse '{impulse_var}'.")
        except Exception as e:
            with cols[plot_idx % n_cols]:
                st.error(f"Error plotting IRF for impulse '{impulse_var}': {e}")
        plot_idx += 1

# Generates a word cloud per topic for topic trend analysis
def generate_wordcloud(words, title):
    text = ' '.join(words)
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def generate_causality_network(
    df_stationary,
    var_results,
    target_variable_name: str,    # Column name of the target variable in df_stationary
    domain_feature_names: list,   # List of domain feature column names in df_stationary
    domain_name_str: str,         # User-friendly name of the domain (for labels)
    sector_name_str: str,         # User-friendly name of the sector (for labels)
    significance_level: float = 0.05,
    weak_significance_level: float = 0.10,
    show_p_values_on_edges: bool = True
):

    pairing_label = f"{domain_name_str} -> {sector_name_str}"

    if not isinstance(df_stationary, pd.DataFrame) or df_stationary.empty:
        print("Error: df_stationary is not a valid or non-empty DataFrame.")
        return None, None
        
    if target_variable_name not in df_stationary.columns:
        print(f"Error: Target variable '{target_variable_name}' not found in df_stationary columns.")
        return None, None
    
    # Ensure domain features are a subset of df_stationary.columns and not the target
    valid_domain_features_for_gc = [
        f for f in domain_feature_names if f in df_stationary.columns and f != target_variable_name
    ]
    if not valid_domain_features_for_gc:
        print(f"Warning: No valid 'causing' domain features found in df_stationary for GC tests against '{target_variable_name}'.")

    # Prepare Graphviz Diagram
    dot = graphviz.Digraph(
        comment=f'Granger Causality Network: {pairing_label}',
        graph_attr={'rankdir': 'LR', 'splines': 'true', 'overlap': 'false', 'fontsize': '10', 'label': f'Causality: {domain_name_str} on {sector_name_str}'},
        node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontsize': '9'},
        edge_attr={'fontsize': '8'}
    )

    all_vars_in_fitted_model = var_results.model.endog_names # Should be df_stationary.columns

    for var_name in all_vars_in_fitted_model:
        if var_name == target_variable_name:
            dot.node(var_name, label=f"{var_name}\n({sector_name_str})", fillcolor='lightblue', color='dodgerblue')
        elif var_name in domain_feature_names: # Check against original input list for styling
            dot.node(var_name, label=f"{var_name}\n({domain_name_str})", fillcolor='lightgoldenrodyellow', color='goldenrod')
        else: # Other variables that were in df_stationary but not target or specified domain features
             dot.node(var_name, label=var_name, fillcolor='lightgrey')

    # Perform & Add Individual Granger Causality Edges (Domain Features -> Target)
    if valid_domain_features_for_gc: # Only if there are features to test
        for causing_feature in valid_domain_features_for_gc:
            try:
                gc_test = var_results.test_causality(
                    caused=target_variable_name,
                    causing=[causing_feature], # Test one at a time
                    kind='f'
                )
                p_value = gc_test.pvalue
                edge_label = f"p={p_value:.3f}" if show_p_values_on_edges else ""
                tooltip_text = f"{causing_feature} GC-> {target_variable_name} (p={p_value:.3f})"

                if p_value < significance_level:
                    dot.edge(causing_feature, target_variable_name, label=edge_label, color="forestgreen", penwidth="2.0", tooltip=tooltip_text)
                elif p_value < weak_significance_level:
                    dot.edge(causing_feature, target_variable_name, label=edge_label, color="orange", style="dashed", penwidth="1.5", tooltip=tooltip_text)
            except Exception as e:
                print(f"Error in GC test for {causing_feature} -> {target_variable_name}: {e}")
    else:
        print(f"No valid domain features specified to test Granger causality against {target_variable_name}.")


    # Add Joint Granger Causality Info as a graph label
    if valid_domain_features_for_gc:
        try:
            joint_gc_test = var_results.test_causality(
                caused=target_variable_name,
                causing=valid_domain_features_for_gc, # Test all valid domain features jointly
                kind='f'
            )
            joint_p_value = joint_gc_test.pvalue
            joint_label_suffix = ""
            if joint_p_value < significance_level: joint_label_suffix = " (Strongly Significant)"
            elif joint_p_value < weak_significance_level: joint_label_suffix = " (Suggestive)"
            else: joint_label_suffix = " (Not Significant)"
            
            current_graph_label = dot.graph_attr.get('label', '')
            dot.graph_attr['label'] = f"{current_graph_label}\nJoint GC ({len(valid_domain_features_for_gc)} field features -> target): p={joint_p_value:.3f}{joint_label_suffix}"
            dot.graph_attr['labelloc'] = 'b' # Bottom label
            
        except Exception as e:
            print(f"Error in Joint GC test for {domain_name_str} features -> {target_variable_name}: {e}")
    else:
        current_graph_label = dot.graph_attr.get('label', '')
        dot.graph_attr['label'] = f"{current_graph_label}\nNo domain features for Joint GC test."
        dot.graph_attr['labelloc'] = 'b'

    return dot

def bsts_forecast_plot(df, train_df,test_df, target_col, prediction_df,scalers):
    """
    Visualizes and evaluates BSTS forecast results.

    Parameters:
    - df: Full unscaled DataFrame with datetime index.
    - test_df: Unscaled test set DataFrame.
    - target_col: Name of the target variable (str).
    - prediction_df: DataFrame returned by Orbit's model.predict()

    Returns:
    - fig: matplotlib figure
    - metrics: DataFrame with MAE, MSE, RMSE
    """
    # Align predictions
    #prediction_df = prediction_df.set_index('date')
    #prediction_df.index = test_df.index[:len(prediction_df)]
    #prediction_df.index = test_df.index[:4]
    prediction_unscaled = prediction_df.copy()
    
    prediction_unscaled['prediction_5']=scalers[target_col].inverse_transform(prediction_df['prediction_5'].values.reshape(-1, 1))
    prediction_unscaled['prediction']=scalers[target_col].inverse_transform(prediction_df['prediction'].values.reshape(-1, 1))
    prediction_unscaled['prediction_95']=scalers[target_col].inverse_transform(prediction_df['prediction_95'].values.reshape(-1, 1))

    test_unscaled = test_df.copy()
    test_unscaled[target_col]=scalers[target_col].inverse_transform(test_df[[target_col]])
    
    #prediction_unscaled["month"] = pd.to_datetime(prediction_unscaled["month"])
    print(prediction_unscaled)
    
    #prediction_unscaled = prediction_unscaled.set_index('month')
    #print(prediction_df)

    forecast_values = prediction_unscaled['prediction']
    actual_values = test_unscaled[target_col]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df[target_col], label='Actual', color='blue', marker='o')
    ax.plot(prediction_unscaled["month"], prediction_unscaled['prediction'], label='Forecast', color='orange', linestyle='--', marker='o')

    # Optional: Add confidence interval if available
    if 'prediction_5' in prediction_unscaled.columns and 'prediction_95' in prediction_unscaled.columns:
        ax.fill_between(prediction_unscaled["month"],
                        prediction_unscaled['prediction_5'],
                        prediction_unscaled['prediction_95'],
                        color='orange', alpha=0.2, label='90% CI')

    ax.axvline(x=prediction_unscaled.iloc[0]['month'], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title(f'Forecast vs Actual: {target_col}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    ax.legend()
    plt.xticks(rotation=90)
    fig.tight_layout()

    # Evaluation
    mae = mean_absolute_error(actual_values, forecast_values)
    mse = mean_squared_error(actual_values, forecast_values)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual_values, forecast_values)*100

    metrics = pd.DataFrame([{
        'MAE': round(mae, 3),
        'MSE': round(mse, 3),
        'RMSE': round(rmse, 3),
        'MAPE (%)': round(mape, 3)
    }])

    return fig, metrics
