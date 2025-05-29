import streamlit as st
import pandas as pd
import utils.var_analysis_functions as var_analysis
import utils.visualization_functions as viz
from utils.useful_functions import scale_data,transform_data
from utils.bsts_analysis_functions import fit_bsts_model

#field_sector_analysis runs the var analysis in the Streamlit app. After fetching the metadata and marketdata from the AWS server, it performs a 1-1 analysis
def trend_analysis_var(
    analysis_type,
    feature_type,
    metadata,
    rolling_metadata,
    topic_words:None,
    marketdata,
    field:str,
    sector:str,
    max_lags_to_test = 5
) -> None:

  if (analysis_type == "field"):
      
      subheader = "Scientific Field Trends vs Sector Price Movement"
      # Filter metadata to get specific field
      metadata_filt = metadata[metadata["field"] == field].reset_index(drop=True)
      rolling_metadata_filt = rolling_metadata[metadata["field"] == field].reset_index(drop=True)
    
      # Filter marketdata to get specific market
      marketdata_filt = marketdata[marketdata["sector"] == sector].reset_index(drop=True)
      
      if feature_type == "Raw":
        sector_return_col = 'avg_close_price'
        field_features = ['num_publications', 'unique_authors', 'avg_diversity_factor']
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
      elif feature_type == "Monthly % Change":
          sector_return_col = "price_change"
          field_features = ["num_publications_pct_change", "unique_authors_pct_change", "avg_diversity_factor_pct_change"]
          #Drop unneeded columns for this analysis
          metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
          marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
    
      elif feature_type == "Rolling Average":
        metadata_filt = rolling_metadata_filt
        sector_return_col = "price_change"
        field_features = ["num_publications_rolling_avg", "unique_authors_rolling_avg", "avg_diversity_factor_rolling_avg"]
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
        pass
    
      elif feature_type == "Growth Rate":
        metadata_filt = rolling_metadata_filt
        sector_return_col = "price_change"
        field_features = ["num_publications_growth_rate", "unique_authors_growth_rate", "avg_diversity_factor_growth_rate"]
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
        metadata_filt = metadata_filt.iloc[1:].reset_index(drop=True)
        marketdata_filt = marketdata_filt.iloc[1:].reset_index(drop=True)
        pass
    
      else:
        raise ValueError(f"Unknown setting for `feature_type`: {feature_type}")
  elif (analysis_type == "topic"):
   
   subheader = "Topic Trends vs Sector Price Movement"
   metadata_filt = metadata.drop(columns=['index'])
   rolling_metadata_filt = rolling_metadata.drop(columns=['index'])
   
   #Lowercase column names for uniformity

   # Make column names lowercase
   metadata_filt.columns = [col.lower() for col in metadata_filt.columns]
   rolling_metadata_filt.columns = [col.lower() for col in rolling_metadata_filt.columns]
   
   # Filter marketdata to get specific market
   marketdata_filt = marketdata[marketdata["sector"] == sector].reset_index(drop=True)
   
   if feature_type == "Raw":
     sector_return_col = 'avg_close_price'
     field_features = ['topic_0', 'topic_1', 'topic_2']
     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]

   elif feature_type == "Monthly % Change":
       sector_return_col = "price_change"
       field_features = ['topic_0_pct_change', 'topic_1_pct_change', 'topic_2_pct_change']
       #Drop unneeded columns for this analysis
       metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
       marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
       
   elif feature_type == "Rolling Average":
     metadata_filt = rolling_metadata_filt
     sector_return_col = "price_change"
     field_features = ["topic_0_avg", "topic_1_avg", "topic_2_avg"]
     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
     
     pass

   elif feature_type == "Growth Rate":
     metadata_filt = rolling_metadata_filt
     sector_return_col = "price_change"
     field_features = ["topic_0_growth", "topic_1_growth", "topic_2_growth"]

     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
 
     metadata_filt = metadata_filt.iloc[1:].reset_index(drop=True)
     marketdata_filt = marketdata_filt.iloc[1:].reset_index(drop=True)
     pass
  
   else:
     raise ValueError(f"Unknown setting for `feature_type`: {feature_type}")     
           
  #Drop index column
  #metadata_filt = metadata_filt.drop(columns=['index'])
  #marketdata_filt = marketdata_filt.drop(columns=['index'])

  # Join
  merged_df = pd.merge(metadata_filt, marketdata_filt, on=["month"], how="inner").reset_index(drop=True)
  merged_df = merged_df[merged_df["month"] != "2020-06"]
  var_data = merged_df[merged_df["sector"] == sector].copy()
  
  var_data.set_index("month", inplace=True)
  

  if (analysis_type == "topic"):
      col1, col2, col3 = st.columns(3)
      for col_name, container in zip(topic_words.columns[1:], [col1, col2, col3]):
          with container:
              st.subheader(f"Word Cloud for {col_name}")
              viz.generate_wordcloud(topic_words[col_name], col_name)
  
  
  st.subheader(subheader)

  viz.plot_time_series(merged_df, "month", field_features+[sector_return_col])

  # drop non-numeric or identifier columns like 'sector'
  var_data = var_data[field_features+[sector_return_col]]

  #Preprocess data for var analysis
  var_data_processed,report,diff_counter = var_analysis.preprocess_data_for_var(var_data,field_features,sector_return_col)
  #print(var_data)
  test_size = 4
  train_df = var_data_processed[:-test_size]
  test_df = var_data_processed[-test_size:]
  
  #scale data
  df_processed,scalers= scale_data(train_df, var_data_processed.columns)
  test_df_scaled= transform_data(test_df, var_data_processed.columns, scalers)
    
  #Fit VAR model
  err_message,var_results = var_analysis.fit_var_model(df_processed)
  #Granger causality

  st.subheader("Granger Causality Network")
  
  if (err_message == ""):          
      network_graph = viz.generate_causality_network(
          df_processed,
          var_results,
          sector_return_col,    # Column name of the target variable in df_stationary
          field_features,   # List of domain feature column names in df_stationary
          field,         # User-friendly name of the domain (for labels)
          sector,         # User-friendly name of the sector (for labels)
         )
     
      if network_graph:
        st.graphviz_chart(network_graph)
      else:
          st.warning("Could not generate Granger Causality Network diagram.")
    
      st.subheader("Impulse Response Plots")
      remaining_features = [col for col in field_features if col in df_processed.columns]
      
      viz.display_irf_plots(
          var_results,
          sector_return_col,
          remaining_features,
          title_fontsize = 24,
          label_fontsize = 22,
          tick_fontsize = 10,
          legend_fontsize = 10)
      
      st.subheader("VAR Forecasts (1 Quarter)")
    
      forecast_fig,forecast_metrics = var_analysis.var_forecast_plot(var_results, var_data,df_processed, test_df_scaled, sector_return_col,scalers,diff_counter)
      st.pyplot(forecast_fig)
      st.markdown("### Forecast Performance Metrics")
      
      # Convert to Markdown table string
      header = "| " + " | ".join(forecast_metrics.columns) + " |"
      separator = "| " + " | ".join(["---"] * len(forecast_metrics.columns)) + " |"
      row = "| " + " | ".join(str(v) for v in forecast_metrics.iloc[0]) + " |"

      markdown_table = "\n".join([header, separator, row])
      st.markdown(markdown_table)
      
  else:
    st.text(err_message)


#field_sector_analysis runs in the Streamlit app. After fetching the metadata and marketdata from the AWS server, it performs a 1-1 analysis

def trend_analysis_bsts(
    analysis_type,
    feature_type,
    metadata,
    rolling_metadata,
    topic_words:None,
    marketdata,
    field:str,
    sector:str,
    max_lags_to_test = 5
) -> None:

  if (analysis_type == "field"):
      
      subheader = "Scientific Field Trends vs Sector Price Movement"
      # Filter metadata to get specific field
      metadata_filt = metadata[metadata["field"] == field].reset_index(drop=True)
      rolling_metadata_filt = rolling_metadata[metadata["field"] == field].reset_index(drop=True)
    
      # Filter marketdata to get specific market
      marketdata_filt = marketdata[marketdata["sector"] == sector].reset_index(drop=True)
      
      if feature_type == "Raw":
        sector_return_col = 'avg_close_price'
        field_features = ['num_publications', 'unique_authors', 'avg_diversity_factor']
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
      elif feature_type == "Monthly % Change":
          sector_return_col = "price_change"
          field_features = ["num_publications_pct_change", "unique_authors_pct_change", "avg_diversity_factor_pct_change"]
          #Drop unneeded columns for this analysis
          metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
          marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
    
      elif feature_type == "Rolling Average":
        metadata_filt = rolling_metadata_filt
        sector_return_col = "price_change"
        field_features = ["num_publications_rolling_avg", "unique_authors_rolling_avg", "avg_diversity_factor_rolling_avg"]
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
        pass
    
      elif feature_type == "Growth Rate":
        metadata_filt = rolling_metadata_filt
        sector_return_col = "price_change"
        field_features = ["num_publications_growth_rate", "unique_authors_growth_rate", "avg_diversity_factor_growth_rate"]
        #Drop unneeded columns for this analysis
        metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
        marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
        
        metadata_filt = metadata_filt.iloc[1:].reset_index(drop=True)
        marketdata_filt = marketdata_filt.iloc[1:].reset_index(drop=True)
        pass
    
      else:
        raise ValueError(f"Unknown setting for `feature_type`: {feature_type}")
  elif (analysis_type == "topic"):
   
   subheader = "Topic Trends vs Sector Price Movement"
   metadata_filt = metadata.drop(columns=['index'])
   rolling_metadata_filt = rolling_metadata.drop(columns=['index'])
   
   #Lowercase column names for uniformity

   # Make column names lowercase
   metadata_filt.columns = [col.lower() for col in metadata_filt.columns]
   rolling_metadata_filt.columns = [col.lower() for col in rolling_metadata_filt.columns]
   
   # Filter marketdata to get specific market
   marketdata_filt = marketdata[marketdata["sector"] == sector].reset_index(drop=True)
   
   if feature_type == "Raw":
     sector_return_col = 'avg_close_price'
     field_features = ['topic_0', 'topic_1', 'topic_2']
     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]

   elif feature_type == "Monthly % Change":
       sector_return_col = "price_change"
       field_features = ['topic_0_pct_change', 'topic_1_pct_change', 'topic_2_pct_change']
       #Drop unneeded columns for this analysis
       metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
       marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
       
   elif feature_type == "Rolling Average":
     metadata_filt = rolling_metadata_filt
     sector_return_col = "price_change"
     field_features = ["topic_0_avg", "topic_1_avg", "topic_2_avg"]
     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
     
     pass

   elif feature_type == "Growth Rate":
     metadata_filt = rolling_metadata_filt
     sector_return_col = "price_change"
     field_features = ["topic_0_growth", "topic_1_growth", "topic_2_growth"]

     #Drop unneeded columns for this analysis
     metadata_filt = metadata_filt.loc[:, metadata_filt.columns.isin(field_features+['month','field'])]
     marketdata_filt = marketdata_filt.loc[:, marketdata_filt.columns.isin([sector_return_col]+['month','sector'])]
 
     metadata_filt = metadata_filt.iloc[1:].reset_index(drop=True)
     marketdata_filt = marketdata_filt.iloc[1:].reset_index(drop=True)
     pass
  
   else:
     raise ValueError(f"Unknown setting for `feature_type`: {feature_type}")     
           
  #Drop index column
  #metadata_filt = metadata_filt.drop(columns=['index'])
  #marketdata_filt = marketdata_filt.drop(columns=['index'])

  # Join
  merged_df = pd.merge(metadata_filt, marketdata_filt, on=["month"], how="inner").reset_index(drop=True)
  merged_df = merged_df[merged_df["month"] != "2020-06"]
  bsts_data = merged_df[merged_df["sector"] == sector].copy()
  
  #var_data.set_index("month", inplace=True)
  

  if (analysis_type == "topic"):
      col1, col2, col3 = st.columns(3)
      for col_name, container in zip(topic_words.columns[1:], [col1, col2, col3]):
          with container:
              st.subheader(f"Word Cloud for {col_name}")
              viz.generate_wordcloud(topic_words[col_name], col_name)
  
  
  st.subheader(subheader)

  viz.plot_time_series(merged_df, "month", field_features+[sector_return_col])

  # drop non-numeric or identifier columns like 'sector'
  bsts_data = bsts_data[field_features+[sector_return_col,'month']]

  test_size = 4
  train_df = bsts_data[:-test_size]
  test_df = bsts_data[-test_size:]
  
  #scale data
  train_df_scaled,scalers= scale_data(train_df, field_features+[sector_return_col])
  test_df_scaled= transform_data(test_df, field_features+[sector_return_col], scalers)
  
  model = fit_bsts_model(train_df_scaled,sector_return_col,'month',field_features)
  
  if (isinstance(model, str)):
      st.text(model)
  else:
      
      st.subheader("VAR Forecasts (1 Quarter)")
      
      prediction_df = model.predict(df=test_df_scaled)
      forecast_fig,forecast_metrics = viz.bsts_forecast_plot(bsts_data, train_df_scaled,test_df_scaled, sector_return_col, prediction_df,scalers)
      st.pyplot(forecast_fig)
      st.markdown("### Forecast Performance Metrics")
      
      # Convert to Markdown table string
      header = "| " + " | ".join(forecast_metrics.columns) + " |"
      separator = "| " + " | ".join(["---"] * len(forecast_metrics.columns)) + " |"
      row = "| " + " | ".join(str(v) for v in forecast_metrics.iloc[0]) + " |"

      markdown_table = "\n".join([header, separator, row])
      st.markdown(markdown_table)
