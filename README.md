# DataCleanerPro

## Overview

**DataCleaner-Pro** is an advanced AI-powered data cleaning and analysis platform. Designed for data scientists, analysts, and professionals working with data, this application provides powerful tools to transform raw datasets into actionable insights.

<img width="1342" height="914" alt="image" src="https://github.com/user-attachments/assets/885d425c-460a-496a-8f57-8eb77313e5a3" />



### 🎯 Objectives

* Automate the data cleaning process
* Provide advanced statistical analysis
* Integrate automatic Machine Learning algorithms
* Offer interactive data visualizations
* Provide a complete REST API for integration

---

## ✨ Key Features

### 🧹 Advanced AI Cleaning

* **Automatic duplicate detection** using smart algorithms
* **Missing value handling** with adaptive methods
* **Format standardization** (dates, numbers, text)
* **Outlier management** using four detection methods (IQR, Z-Score, Isolation Forest, LOF)
* **Text normalization** and data type optimization

### 🔄 Simplified AutoETL

* **Multi-format support**: CSV, Excel (.xlsx/.xls), JSON, TSV
* **Automated ETL pipelines** with intelligent transformations
* **Database connectors** (PostgreSQL, MySQL, SQLite, MongoDB)
* **Smart dataset merging** with key detection
* **Multi-format export** with metadata preservation

### 📊 Advanced Analytics

* **Comprehensive descriptive statistics** (mean, median, standard deviation, quartiles)
* **Correlation analysis** with matrices and heatmaps
* **Automated Machine Learning** (classification, regression, clustering)
* **Anomaly detection** and distribution analysis
* **AI-generated insights** with smart recommendations

### 📈 Interactive Visualizations

* **Dynamic charts** using Chart.js and Plotly
* **Customizable real-time dashboards**
* **Trend analysis** and time series tracking
* **High-resolution exports** (PNG, SVG, PDF)
* **3D visualizations** for complex analysis

### 🔗 Complete REST API

* **Full RESTful endpoints** for all features
* **Token-based authentication** with permission control
* **Integrated OpenAPI/Swagger documentation**
* **Rate limiting** and performance monitoring
* **Webhooks** for advanced integrations

---

## 🏗️ Technical Architecture

### Technology Stack

* **Backend**: Flask 2.3.3, SQLAlchemy, Werkzeug
* **Database**: SQLite (dev), PostgreSQL (prod), Redis (cache)
* **Frontend**: Bootstrap 5, Chart.js, Plotly.js, FontAwesome
* **Data Science**: pandas, NumPy, scikit-learn, matplotlib, seaborn
* **Advanced ML**: TensorFlow, PyTorch, XGBoost, LightGBM
* **Visualization**: Plotly, Bokeh, Altair
* **API**: Flask-RESTx, Marshmallow
* **Authentication**: Flask-Login, JWT, OAuth2
* **Monitoring**: Sentry, Prometheus
* **Deployment**: Gunicorn, Docker, Kubernetes

### Project Structure

```
webapp/
├── app.py                     # Main application (8000+ lines)
├── config.py                  # Advanced configuration
├── run.py                     # Startup script
├── requirements.txt           # Dependencies (200+ packages)
├── .gitignore
├── templates/
│   ├── base.html              # Base template with Bootstrap 5
│   ├── index.html             # Home page
│   ├── dashboard.html         # Analytics dashboard
│   ├── upload.html            # Upload interface
│   ├── auth/                  # Authentication
│   ├── datasets/              # Dataset management
│   ├── jobs/                  # Cleaning jobs
│   ├── analysis/              # Analysis results
│   └── errors/                # Error pages
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── uploads/
├── logs/
└── data_processing/
```

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.8+
* pip or conda
* Git
* Node.js (optional for frontend builds)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-username/datacleaner-pro.git
cd datacleaner-pro

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python run.py --init-db

# Run the app
python run.py
```

### Advanced Configuration

#### Environment Variables

```bash
FLASK_ENV=development
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/datacleaner_pro
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_S3_BUCKET=your-bucket
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
SENTRY_DSN=your-sentry-dsn
```

---

## 📊 Implemented Features

### ✅ Authentication & User Management

* Registration and login with validation
* Profile and preference management
* Tiered quotas (Free, Pro, Enterprise)
* OAuth login (Google, LinkedIn)
* Token-based API authentication

### ✅ Dataset Management

* Drag-and-drop uploads with progress bar
* Multi-format support (CSV, Excel, JSON, TSV)
* Automatic encoding detection
* Metadata extraction (rows, columns, types)
* Version control and backup

### ✅ Automatic Cleaning

* 4 cleaning modes (auto, conservative, aggressive, custom)
* Duplicate detection and removal
* 8 missing-value handling methods
* Outlier detection via multiple algorithms
* Format standardization & text normalization
* Automatic type optimization

### ✅ Statistical Analysis

* Complete descriptive statistics
* Correlation analysis with heatmaps
* Normality and skewness tests
* Distribution & anomaly analysis
* Automated data profiling

### ✅ AutoML

* Automated classification (Random Forest, XGBoost)
* Regression with performance metrics
* K-Means clustering with auto-optimization
* Feature importance analysis
* Cross-validation and metrics reporting

### ✅ Advanced Visualization

* Interactive charts (Chart.js, Plotly)
* Histograms and boxplots
* Correlation matrices
* PCA-based clustering plots
* High-resolution exports

### ✅ REST API

* Full-featured endpoints
* Token-based authentication
* Auto-generated Swagger docs
* Rate limiting & monitoring
* Proper error handling

### ✅ Analytics Dashboard

* Real-time metrics (datasets, jobs, quota)
* 30-day activity charts
* AI-powered recommendations
* Task history tracking
* Quick actions and user settings

---

## 🛠️ Advanced Features

### Integrated Artificial Intelligence

```python
from app import DataCleaner, MLAnalyzer

# Automated cleaning
cleaner = DataCleaner()
cleaned_df, report = cleaner.clean_dataset(df, {
    'remove_duplicates': True,
    'handle_missing_values': 'auto',
    'detect_outliers': True,
    'outlier_method': 'isolation_forest',
    'standardize_formats': True
})

# Automated ML analysis
analyzer = MLAnalyzer()
results = analyzer.auto_analyze(cleaned_df, target_column='target', analysis_type='classification')
```

### Automatic Insight Generation

```python
from app import InsightGenerator

generator = InsightGenerator()
insights = generator.generate_insights(analysis_results, df)

# Example insights:
# - Data quality: "Excellent, with 95% completeness"
# - Correlations: "Strong correlation between 'age' and 'income'"
# - ML Performance: "Classification model achieved 87% accuracy"
```

---

## 🔧 Next Steps

### Features to Develop

1. **Advanced ETL Pipelines**

   * Kafka & Redis Streams connectors
   * Real-time transformation
   * Airflow orchestration

2. **ML Ops Integration**

   * Model versioning (MLflow)
   * Auto-deployment
   * Data drift monitoring

3. **Advanced Analytics**

   * Time series with Prophet
   * NLP and sentiment analysis
   * Computer vision support

4. **Collaboration Tools**

   * Shared workspaces
   * Comments & annotations
   * Approval workflows

5. **Cloud Integrations**

   * AWS SageMaker
   * Google Cloud AI
   * Azure Machine Learning

---

## 📈 Project Stats

| Feature              | Status | Coverage |
| -------------------- | ------ | -------- |
| Authentication       | ✅ Done | 100%     |
| Dataset Upload       | ✅ Done | 100%     |
| AI Cleaning          | ✅ Done | 95%      |
| Statistical Analysis | ✅ Done | 90%      |
| Machine Learning     | ✅ Done | 85%      |
| Visualizations       | ✅ Done | 80%      |
| REST API             | ✅ Done | 90%      |
| Dashboard            | ✅ Done | 95%      |
| Documentation        | ✅ Done | 100%     |

---

**DataCleaner-Pro** — Transform your data into a competitive advantage 🚀
*Developed with ❤️ for the Data Science community*

---
