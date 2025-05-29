# Literature Analytics Platform

A sophisticated analytical platform that explores correlations between scientific literature trends and stock market movements using advanced time series analysis.

## 🎯 Project Overview

This platform investigates whether **scientific literature trends** can help **predict short-term stock price movements** in specific market sectors. By analyzing publication patterns from arXiv alongside market data from Yahoo Finance, we apply Vector Autoregression (VAR) and Bayesian Structural Time Series (BSTS) models to uncover potential predictive relationships.

## 🏗️ Architecture

```
Literature-Analytics-Platform/
├── data_collection/
│   ├── metadata/
│   │   └── metadata_collection.py      # ArXiv data scraper
│   └── marketdata/
│       ├── marketdata_collection.py    # Yahoo Finance data collector
│       └── marketdata_collection.ipynb # Jupyter notebook version
├── processing/
│   └── metadata_processing.py          # Feature extraction & LDA modeling
├── pages/
│   ├── field_trend_var_analysis.py     # VAR analysis for field trends
│   ├── field_trend_bsts_analysis.py    # BSTS analysis for field trends
│   ├── topic_trend_var_analysis.py     # VAR analysis for topic trends
│   └── topic_trend_bsts_analysis.py    # BSTS analysis for topic trends
├── utils/
│   ├── arXiv_scraper_functions.py     # ArXiv API utilities
│   ├── field_growth_processing_functions.py
│   ├── topic_growth_processing_functions.py
│   ├── streamlit_functions.py         # Analysis and visualization
│   └── db_manager.py                   # Database connection manager
├── sample_data/                        # Sample datasets for testing
├── main.py                            # Streamlit application entry point
├── config.py                          # Configuration management
├── Dockerfile                         # Docker configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
└── README.md                          # This file
```

## 🚀 Features

- **Real-time Data Collection**: Automated scraping from arXiv API and Yahoo Finance
- **Advanced NLP Processing**: Topic modeling using Latent Dirichlet Allocation (LDA)
- **Time Series Analysis**: VAR and BSTS models for trend prediction
- **Interactive Dashboards**: Streamlit-based web interface for analysis
- **Cloud Infrastructure**: AWS EC2 hosted MySQL database
- **Containerized Deployment**: Docker support for easy deployment

## 📊 Analysis Metrics

### Field Growth Metrics
- Number of publications per field
- Unique authors count
- Diversity factor (cross-disciplinary collaboration)

### Topic Modeling
- LDA-based topic extraction from abstracts
- Monthly topic trend analysis
- Topic-market sector correlation

### Market Analysis
- ETF price movements for 5 major sectors
- Monthly percentage changes
- Correlation with literature trends

## 🛠️ Technical Stack

- **Backend**: Python 3.10
- **Database**: MySQL 8.0 (AWS RDS)
- **Web Framework**: Streamlit
- **Data Analysis**: pandas, numpy, statsmodels
- **Machine Learning**: gensim (LDA), orbit-ml (BSTS)
- **Visualization**: plotly, matplotlib, wordcloud
- **Infrastructure**: Docker, AWS EC2

## 📋 Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Python 3.10+ (for local development)
- MySQL 8.0+ (optional for local database)
- Git

## 🔧 Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Iska1999/Literature-Analytics-Platform.git
   cd Literature-Analytics-Platform
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Build and run with Docker**
   ```bash
   docker build -t lit_platform .
   docker run -p 8501:8501 --env-file .env lit_platform
   ```

4. **Access the application**
   ```
   http://localhost:8501
   ```

### Option 2: Local Development

1. **Clone and navigate to the repository**
   ```bash
   git clone https://github.com/Iska1999/Literature-Analytics-Platform.git
   cd Literature-Analytics-Platform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Run data collection (optional - sample data provided)**
   ```bash
   python data_collection/metadata/metadata_collection.py
   python data_collection/marketdata/marketdata_collection.py
   ```

6. **Process the data**
   ```bash
   python processing/metadata_processing.py
   ```

7. **Launch the application**
   ```bash
   streamlit run main.py
   ```

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file based on `.env.example`:

```bash
# Database Configuration
DB_HOST=your_mysql_host
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=literature_analytics_platform
```

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

## 📈 Usage

1. **Select Analysis Type**: Choose from the sidebar:
   - Field Trend VAR Analysis
   - Field Trend BSTS Analysis
   - Topic Trend VAR Analysis
   - Topic Trend BSTS Analysis

2. **Configure Parameters**:
   - Feature Type: Raw, Monthly % Change, Rolling Average, or Growth Rate
   - Scientific Field: Computer Science, Mathematics, Physics, etc.
   - Market Sector: Technology, Healthcare, Financials, Energy, or Industrials

3. **Run Analysis**: Click the "🔍 Run Analysis" button to generate results

4. **View Results**: Interactive charts and statistical outputs will be displayed

## 🗄️ Database Schema

The platform uses the following main tables:

- `monthly_metadata`: Aggregated monthly metrics by field
- `rolling_metadata`: Rolling averages for smoothed trends
- `monthly_topic`: LDA topic distributions by month
- `rolling_topic`: Smoothed topic trends
- `topic_words`: Top words for each LDA topic
- `marketdata`: ETF prices and percentage changes

## 🧪 Sample Data

Sample processed data (collected up to 05/22/2025) is provided in the `sample_data/` directory for testing without running the full data collection pipeline.

## 🐛 Troubleshooting

### Database Connection Issues
- Verify your `.env` file contains correct credentials
- Ensure your IP is whitelisted in AWS security groups
- Check network connectivity to the database host

### Memory Issues
- The LDA model can be memory-intensive. Consider reducing the dataset size or using a machine with more RAM
- Use Docker memory limits if needed: `docker run -m 4g ...`

### Missing Dependencies
- Ensure all system dependencies are installed (especially for Docker)
- Try clearing pip cache: `pip cache purge`

## 📊 Performance Considerations

- Data is processed in monthly batches to optimize memory usage
- Database queries are cached in Streamlit session state
- LDA model parameters are tuned for balance between accuracy and performance
- Connection pooling is handled by SQLAlchemy

## 🔒 Security Notes

- Database credentials are stored in environment variables
- Never commit sensitive information to version control
- Use HTTPS in production deployments
- Regularly update dependencies for security patches

## 🚦 Limitations & Disclaimers

- This is a research prototype for technical assessment purposes
- Should NOT be used as financial or trading advice
- Historical correlations don't guarantee future performance
- Further model tuning and validation would be needed for production use

## 🔮 Future Enhancements

- [ ] Implement real-time streaming data pipeline
- [ ] Add more sophisticated NLP models (BERT, GPT)
- [ ] Expand to additional data sources (PubMed, RePEc)
- [ ] Implement user authentication and personalization
- [ ] Add automated model retraining
- [ ] Create API endpoints for programmatic access
- [ ] Implement A/B testing for model comparison

## 🤝 Contributing

This project was created as part of a technical assessment. For any questions or suggestions, please feel free to reach out.

## 📄 License

[Specify your license here]

## 👤 Author

**[Your Name]**
- GitHub: [@Iska1999](https://github.com/Iska1999)
- Email: [Your Email]

---

*Developed for the second round technical assessment @Ghamut*