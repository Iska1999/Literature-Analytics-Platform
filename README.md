# Literature-Analytics-Platform

A basic analytical platform that explores correlations between scientific literature trends and stock market movements using time series analysis.

## Overview

This platform investigates whether **scientific literature trends** can help **predict short-term stock price movements** in specific market sectors. By analyzing publication patterns from arXiv alongside market data from Yahoo Finance, we apply Vector Autoregression (VAR) and Bayesian Structural Time Series (BSTS) models to uncover potential predictive relationships.

---

## Contents

- [`data_collection/`](data_collection): Contains the scripts to scrape the meta- and marketdata, each in separate folders.

- [`pages/`](pages): Contains Python scripts for the Streamlit application pages.

- [`processing/`](processing): Contains the Python script to process the metadata and extract the required features for analysis.

- [`sample_data/`](sample_data): Contains sample processed meta- and market data collected up to 05/22/2025.

- [`utils/`](utils): Contains utility functions used by the scripts.

- [`main.py`](main.py): The main page for the Streamlit application.
  
- [`config.py`](config.py): The Python file with configuration management information.

- [`Dockerfile`](Dockerfile): The dockerfile to run the application on a docker platform.

- [`Literature-Analytics-Platform-Presentation.pdf`](Literature-Analytics-Platform-Presentation.pdf): a presentation of the work provided in this repository.

- [`requirements.txt`](requirements.txt): The prerequisites necessary to run the interactive web application and processing functions.


## Installation

### Prerequisites

- Docker
- Git
- Python 3.10+ (for local development)
- MySQL 8.0+ (optional for local database)

Follow the steps of Option 1 if interested in only seeing the interactive web application based on the features already uploaded to the SQL server. Follow the steps of Option 2 if interested in collecting, processing, and uploading the features.

### Option 1: Docker Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Iska1999/Literature-Analytics-Platform.git
   cd Literature-Analytics-Platform
   ```

2. **Build and run with Docker**
   ```bash
   docker build -t lit_platform .
   docker run -p 8501:8501 lit_platform
   ```

3. **Access the application by pasting this URL in your browser**
   ```
   http://localhost:8501
   ```

### Option 2: Local Development

1. **Clone and navigate to the repository**
   ```bash
   git clone https://github.com/Iska1999/Literature-Analytics-Platform.git
   cd Literature-Analytics-Platform
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data collection (optional - sample data provided)**
   ```bash
   python data_collection/metadata/metadata_collection.py
   python data_collection/marketdata/marketdata_collection.py
   ```

4. **Process the data**
   ```bash
   python processing/metadata_processing.py
   ```

5. **Launch the application**
   ```bash
   streamlit run main.py
   ```

## Analysis Metrics

### Field Growth Metrics
- Number of publications per field
- Unique authors count
- Diversity factor (cross-disciplinary collaboration)

### Topic Growth Metrics
- Monthly averages of top 3 latent topics (topics were generated using Latent Dirichlet Allocation (LDA))

### Market Analysis
- ETF price movements for 5 major sectors
- Monthly percentage changes

## Usage

1. **Select Analysis Type**: Choose from the sidebar:
   - Field Trend VAR Analysis
   - Field Trend BSTS Analysis
   - Topic Trend VAR Analysis
   - Topic Trend BSTS Analysis

2. **Configure Parameters**:
   - Feature Type: Raw, Monthly % Change, Rolling Average, or Growth Rate
   - Scientific Field: Computer Science, Mathematics, Physics, Economics, Electrical Engineering & Systems Science, and Quantitative Biology
   - Market Sector: Technology, Healthcare, Financials, Energy, or Industrials

3. **Run Analysis**: Click the "🔍 Run Analysis" button to generate results

4. **View Results**: Interactive charts and statistical outputs will be displayed

## 🗄️ Database Schema

The platform uses the following main tables:

- `monthly_metadata`: Aggregated monthly metrics by scientific field
- `rolling_metadata`: Rolling averages for smoothed trends (period = 6 months)
- `monthly_topic`: LDA topic distributions by month
- `rolling_topic`: Smoothed topic trends (period = 6 months)
- `topic_words`: Top words for each LDA topic
- `marketdata`: ETF prices and percentage changes

## Future Enhancements

- [ ] Collect more data
- [ ] Analyze using LSTM and CNNs (assuming more data is obtained)
- [ ] Expand to additional data sources (PubMed, Wiley etc.)
- [ ] Implement user authentication and personalization
- [ ] Implement rolling forecast
- [ ] Fine-tune LDA and BSTS models
- [ ] Implement A/B testing for model comparison

## Memory Issues

The arXiv metadata scraping and LDA model training can be time-consuming (around 17 hours) and memory-intensive. Consider reducing the dataset size or using a machine with more RAM.

## Disclaimer

This project was undertaken as part of a technical assessment. As a result, several significant improvements (i.e., model finetuning, different forecasting methods, more data) are yet to be made. Do NOT use this platform as a financials or trading advisor.

*Developed for the second round technical assessment @Ghamut*
