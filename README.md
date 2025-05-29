# Literature-Analytics-Platform

This project investigates whether **scientific literature trends** can help **predict short-term stock price movements** in specific market sectors.

Using data from **arXiv** and **Yahoo Finance**, we process and model monthly metrics with (a) **Vector Autoregression (VAR)** and (b) **Bayesian Structural Time Series (BSTS)** frameworks to explore interactions between scientific literature trends and short-term stock price movements.

---

## Contents

[`data_collection/`](data_collection): Contains the scripts to scrape the meta- and marketdata, each in separate folders.

[`pages/`](pages): Contains Python scripts for the Streamlit application pages.

[`processing/`](processing): Contains the Python script to process the metadata and extract the required features for analysis.

[`sample_data/`](sample_data): Contains sample processed meta- and market data collected up to 05/22/2025.

[`utils/`](utils): Contains utility functions used by the scripts.

[`main.py`](main.py): The main page for the Streamlit application.

[`dockerfile`](dockerfile): The dockerfile to run the application on a docker platform.

[`requirements.txt`](requirements.txt): The prerequisites necessary to run the interactive web application and processing functions.

## Installation

Accessing the interactive web application requires two steps: 

### 1. Clone the repository

Open your terminal (Command Prompt, PowerShell, or terminal app) and run:

```shell
git clone https://github.com/Iska1999/Literature-Analytics-Platform.git
```
Next, navigate to the repository folder using:

```shell
cd Literature-Analytics-Platform
```

### 2. Build the Docker image

Use [Docker](https://www.docker.com/) or an online container deployment platform like [render.com](https://www.render.com) to run the dockerfile with the necessary prerequisites found in the [`requirements.txt`](requirements.txt).
Run the following command from the root of the repo:

```shell
docker build -t lit_platform .
```
### 3. Run the Docker container

```shell
docker run -p 8501:8501 lit_platform
```

Alternatively, if only interested in quickly testing the interactive web application without installing via Docker, you can check out this deployed version on [render.com](https://literature-analytics-platform.onrender.com/topic_trend_analysis). Kindly note that bootup time might take more than 50 seconds since it is deployed on a Free Tier Plan.

**NOTE:** the interactive web application uses an AWS EC2 server instance to fetch the processed features and perform the analysis in real-time. The IP address and credentials are currently hard-coded in this prototype, but to setup your own AWS EC2 server, kindly follow these [instructions](https://www.geeksforgeeks.org/amazon-ec2-creating-an-elastic-cloud-compute-instance/), and modify the hard-coded IP addresses in the [`pages/`](pages) scripts.

## Disclaimer

This project was undertaken as part of a technical assessment. As a result, several significant improvements (i.e., model finetuning, different forecasting methods, more data) are yet to be made. Do NOT use this platform as a financials or trading advisor.
