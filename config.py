"""
Configuration settings for DataCleaner-Pro application.
"""

import os
from datetime import timedelta

# Base directory of the application
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration class with common settings."""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'datacleaner_pro.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_timeout': 20,
        'pool_recycle': -1,
        'pool_pre_ping': True,
        'connect_args': {'timeout': 30} if 'sqlite' in os.environ.get('DATABASE_URL', '') else {}
    }
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'tsv'}
    
    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_HEADERS_ENABLED = True
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Data processing settings
    MAX_ROWS_PREVIEW = 1000
    MAX_COLUMNS_ANALYSIS = 50
    ASYNC_PROCESSING_ENABLED = True
    
    # Pagination settings
    DATASETS_PER_PAGE = 12
    JOBS_PER_PAGE = 20
    ANALYSES_PER_PAGE = 15
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(basedir, 'logs', 'datacleaner_pro.log')
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Security settings
    BCRYPT_LOG_ROUNDS = 12
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # API settings
    API_TITLE = 'DataCleaner-Pro API'
    API_VERSION = 'v1'
    OPENAPI_VERSION = '3.0.2'
    
    # Cloud storage settings (optional)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Google Cloud settings (optional)
    GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CLOUD_BUCKET = os.environ.get('GOOGLE_CLOUD_BUCKET')
    
    # Azure settings (optional)
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME')
    
    # OAuth settings
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    
    GITHUB_CLIENT_ID = os.environ.get('GITHUB_CLIENT_ID')
    GITHUB_CLIENT_SECRET = os.environ.get('GITHUB_CLIENT_SECRET')
    
    LINKEDIN_CLIENT_ID = os.environ.get('LINKEDIN_CLIENT_ID')
    LINKEDIN_CLIENT_SECRET = os.environ.get('LINKEDIN_CLIENT_SECRET')
    
    # Monitoring settings
    SENTRY_DSN = os.environ.get('SENTRY_DSN')
    
    # Feature flags
    ENABLE_PREMIUM_FEATURES = os.environ.get('ENABLE_PREMIUM_FEATURES', 'true').lower() in ['true', '1']
    ENABLE_ML_FEATURES = os.environ.get('ENABLE_ML_FEATURES', 'true').lower() in ['true', '1']
    ENABLE_API_ACCESS = os.environ.get('ENABLE_API_ACCESS', 'true').lower() in ['true', '1']
    
    # Default user quotas
    DEFAULT_FREE_QUOTA_MB = 1000  # 1GB
    DEFAULT_PREMIUM_QUOTA_MB = 100 * 1024  # 100GB
    DEFAULT_ENTERPRISE_QUOTA_MB = 1000 * 1024  # 1TB
    
    # Performance settings
    CHUNK_SIZE = 10000  # For processing large datasets
    MAX_WORKERS = 4  # For parallel processing
    
    # ML Model settings
    MODEL_CACHE_DIR = os.path.join(basedir, 'models')
    ENABLE_MODEL_CACHING = True
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration."""
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.dirname(app.config['LOG_FILE']), exist_ok=True)
        os.makedirs(app.config.get('MODEL_CACHE_DIR', ''), exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development database (SQLite)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'datacleaner_pro_dev.db')
    
    # Relaxed security for development
    WTF_CSRF_ENABLED = True
    BCRYPT_LOG_ROUNDS = 4  # Faster for development
    
    # Development mail settings (console backend)
    MAIL_SUPPRESS_SEND = False
    MAIL_DEBUG = True
    
    # Cache settings for development
    CACHE_TYPE = 'simple'
    
    # Feature flags for development
    ENABLE_PREMIUM_FEATURES = True
    ENABLE_ML_FEATURES = True
    ENABLE_API_ACCESS = True
    
    # Smaller quotas for development
    DEFAULT_FREE_QUOTA_MB = 100  # 100MB
    DEFAULT_PREMIUM_QUOTA_MB = 10 * 1024  # 10GB

class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING = True
    WTF_CSRF_ENABLED = False
    
    # In-memory SQLite for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable mail sending during tests
    MAIL_SUPPRESS_SEND = True
    
    # Disable rate limiting for tests
    RATELIMIT_ENABLED = False
    
    # Smaller quotas for testing
    DEFAULT_FREE_QUOTA_MB = 10  # 10MB
    DEFAULT_PREMIUM_QUOTA_MB = 100  # 100MB
    
    # Fast password hashing for tests
    BCRYPT_LOG_ROUNDS = 1

class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production database (PostgreSQL recommended)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@localhost/datacleaner_pro'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # SSL redirect
    PREFERRED_URL_SCHEME = 'https'
    
    # Production cache (Redis recommended)
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379'
    
    # Rate limiting with Redis
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379'
    
    # Production mail settings
    MAIL_SUPPRESS_SEND = False
    
    # Higher security settings
    BCRYPT_LOG_ROUNDS = 15
    
    # Production performance settings
    MAX_WORKERS = 8
    CHUNK_SIZE = 50000
    
    @classmethod
    def init_app(cls, app):
        """Production-specific initialization."""
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)

class DockerConfig(ProductionConfig):
    """Docker container configuration."""
    
    # Container-specific settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://postgres:password@db:5432/datacleaner_pro'
    
    CACHE_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://redis:6379'
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://redis:6379'
    
    @classmethod
    def init_app(cls, app):
        """Docker-specific initialization."""
        ProductionConfig.init_app(app)
        
        # Log to stdout for container environments
        import logging
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)

class HerokuConfig(ProductionConfig):
    """Heroku deployment configuration."""
    
    # Heroku-specific settings
    SSL_REDIRECT = True
    
    @classmethod
    def init_app(cls, app):
        """Heroku-specific initialization."""
        ProductionConfig.init_app(app)
        
        # Handle proxy headers
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
        
        # Log to stdout
        import logging
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'heroku': HerokuConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class by name."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, DevelopmentConfig)