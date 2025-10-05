"""
DataCleaner-Pro - Application Flask Professionnelle
===================================================
Application complète de nettoyage et d'analyse de données avec IA intégrée.

Fonctionnalités principales:
- Upload et gestion de datasets (CSV, Excel, JSON)
- Nettoyage automatique intelligent avec IA
- Pipeline ETL simplifié
- Analyses statistiques complètes
- Machine Learning automatique
- Visualisations interactives
- API REST pour intégration
- Interface moderne et responsive

Author: DataCleaner-Pro Team
Version: 2.0.0
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps
import secrets
from io import BytesIO
import base64

# Flask imports
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, jsonify, send_file, session, abort, current_app
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, 
    login_required, current_user
)
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_mail import Mail, Message
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# WTForms imports
from wtforms import (
    StringField, PasswordField, SubmitField, TextAreaField, 
    SelectField, BooleanField, IntegerField, FloatField, 
    FileField as WTFormsFileField, HiddenField
)
from wtforms.validators import (
    DataRequired, Email, Length, EqualTo, NumberRange,
    ValidationError, Optional
)

# Data Science imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

    
# Configuration de l'application
class Config:
    """Configuration de base de l'application."""
    
    # Sécurité
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    WTF_CSRF_TIME_LIMIT = 3600
    
    # Base de données
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///datacleaner_pro.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_timeout': 20,
        'pool_recycle': -1,
        'pool_pre_ping': True
    }
    
    # Upload des fichiers
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'tsv'}
    
    # Mail configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.googlemail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Cache configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    
    # Data processing
    MAX_ROWS_PREVIEW = 1000
    MAX_COLUMNS_ANALYSIS = 50
    ASYNC_PROCESSING_ENABLED = True
    
    # Pagination
    DATASETS_PER_PAGE = 12
    JOBS_PER_PAGE = 20
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = 'logs/datacleaner_pro.log'

class DevelopmentConfig(Config):
    """Configuration pour le développement."""
    DEBUG = True
    LOG_LEVEL = logging.DEBUG

class ProductionConfig(Config):
    """Configuration pour la production."""
    DEBUG = False
    LOG_LEVEL = logging.WARNING
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

# Sélection de la configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Initialisation des extensions Flask
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
mail = Mail()
cache = Cache()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

# Configuration du gestionnaire de connexions
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    """Charge un utilisateur par son ID."""
    return User.query.get(int(user_id))



def inject_user():
    return dict(user=current_user)
    
# Modèles de base de données
class User(UserMixin, db.Model):
    """Modèle utilisateur avec fonctionnalités avancées."""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    
    # Statut et métadonnées
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_premium = db.Column(db.Boolean, default=False, nullable=False)
    email_confirmed = db.Column(db.Boolean, default=False, nullable=False)
    
    # Gestion des quotas
    quota_used = db.Column(db.Float, default=0.0, nullable=False)  # En MB
    quota_limit = db.Column(db.Float, default=1000.0, nullable=False)  # En MB
    
    # Dates importantes
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Préférences utilisateur
    default_cleaning_mode = db.Column(db.String(20), default='automatic')
    auto_analysis_enabled = db.Column(db.Boolean, default=True)
    notification_preferences = db.Column(db.Text)  # JSON string
    
    # Relations
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    cleaning_jobs = db.relationship('CleaningJob', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    api_tokens = db.relationship('APIToken', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash et définit le mot de passe."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Vérifie le mot de passe."""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Retourne le nom complet."""
        return f"{self.first_name} {self.last_name}"
    
    def can_upload(self, file_size_mb):
        """Vérifie si l'utilisateur peut uploader un fichier."""
        return (self.quota_used + file_size_mb) <= self.quota_limit
    
    def update_quota(self, size_mb):
        """Met à jour le quota utilisé."""
        self.quota_used += size_mb
        if self.quota_used < 0:
            self.quota_used = 0
        db.session.commit()
    
    def get_quota_percentage(self):
        """Retourne le pourcentage de quota utilisé."""
        return (self.quota_used / self.quota_limit) * 100 if self.quota_limit > 0 else 0
    
    def to_dict(self):
        """Convertit l'utilisateur en dictionnaire."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.get_full_name(),
            'is_premium': self.is_premium,
            'quota_used': self.quota_used,
            'quota_limit': self.quota_limit,
            'quota_percentage': self.get_quota_percentage(),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }

class Dataset(db.Model):
    """Modèle dataset avec métadonnées complètes."""
    
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # En bytes
    
    # Métadonnées du dataset
    rows_count = db.Column(db.Integer)
    columns_count = db.Column(db.Integer)
    memory_usage = db.Column(db.Float)  # En MB
    encoding = db.Column(db.String(20), default='utf-8')
    
    # Statuts et flags
    is_cleaned = db.Column(db.Boolean, default=False, nullable=False)
    is_analyzed = db.Column(db.Boolean, default=False, nullable=False)
    has_duplicates = db.Column(db.Boolean, default=False)
    has_missing_values = db.Column(db.Boolean, default=False)
    duplicates_count = db.Column(db.Integer, default=0)
    missing_values_count = db.Column(db.Integer, default=0)
    
    # Informations sur les colonnes (JSON)
    columns_info = db.Column(db.Text)  # JSON string with column metadata
    data_types = db.Column(db.Text)    # JSON string with data types
    statistics_summary = db.Column(db.Text)  # JSON string with basic stats
    
    # Dates
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_cleaned = db.Column(db.DateTime)
    last_analyzed = db.Column(db.DateTime)
    
    # Relations
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    cleaning_jobs = db.relationship('CleaningJob', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')
    analysis_results = db.relationship('AnalysisResult', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')
    
    def get_file_size_mb(self):
        """Retourne la taille du fichier en MB."""
        return round(self.file_size / (1024 * 1024), 2)
    
    def get_columns_info(self):
        """Retourne les informations des colonnes."""
        if self.columns_info:
            return json.loads(self.columns_info)
        return []
    
    def set_columns_info(self, info):
        """Définit les informations des colonnes."""
        self.columns_info = json.dumps(info, ensure_ascii=False)
    
    def get_data_types(self):
        """Retourne les types de données."""
        if self.data_types:
            return json.loads(self.data_types)
        return {}
    
    def set_data_types(self, types):
        """Définit les types de données."""
        self.data_types = json.dumps(types, ensure_ascii=False)
    
    def get_statistics_summary(self):
        """Retourne le résumé statistique."""
        if self.statistics_summary:
            return json.loads(self.statistics_summary)
        return {}
    
    def set_statistics_summary(self, stats):
        """Définit le résumé statistique."""
        self.statistics_summary = json.dumps(stats, ensure_ascii=False, default=str)
    
    def update_metadata(self, df):
        """Met à jour les métadonnées à partir d'un DataFrame pandas."""
        self.rows_count = len(df)
        self.columns_count = len(df.columns)
        self.memory_usage = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        
        # Détection des doublons
        self.duplicates_count = df.duplicated().sum()
        self.has_duplicates = self.duplicates_count > 0
        
        # Détection des valeurs manquantes
        self.missing_values_count = df.isnull().sum().sum()
        self.has_missing_values = self.missing_values_count > 0
        
        # Types de données
        types_dict = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            types_dict[col] = dtype
        self.set_data_types(types_dict)
        
        # Informations sur les colonnes
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            columns_info.append(col_info)
        self.set_columns_info(columns_info)
        
        self.last_modified = datetime.utcnow()
    
    def to_dict(self):
        """Convertit le dataset en dictionnaire."""
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'file_size_mb': self.get_file_size_mb(),
            'rows_count': self.rows_count,
            'columns_count': self.columns_count,
            'is_cleaned': self.is_cleaned,
            'is_analyzed': self.is_analyzed,
            'has_duplicates': self.has_duplicates,
            'has_missing_values': self.has_missing_values,
            'duplicates_count': self.duplicates_count,
            'missing_values_count': self.missing_values_count,
            'upload_date': self.upload_date.isoformat(),
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'owner': self.owner.username if self.owner else None
        }

class CleaningJob(db.Model):
    """Modèle pour les tâches de nettoyage."""
    
    __tablename__ = 'cleaning_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='pending', nullable=False)  # pending, running, completed, failed
    progress = db.Column(db.Integer, default=0, nullable=False)  # 0-100
    
    # Configuration du nettoyage
    cleaning_mode = db.Column(db.String(20), default='automatic')  # automatic, conservative, aggressive, custom
    remove_duplicates = db.Column(db.Boolean, default=True)
    handle_missing_values = db.Column(db.String(20), default='auto')  # auto, drop, fill_mean, fill_median, fill_mode
    detect_outliers = db.Column(db.Boolean, default=True)
    standardize_formats = db.Column(db.Boolean, default=True)
    
    # Résultats du nettoyage
    original_rows = db.Column(db.Integer)
    cleaned_rows = db.Column(db.Integer)
    removed_duplicates = db.Column(db.Integer, default=0)
    filled_missing_values = db.Column(db.Integer, default=0)
    detected_outliers = db.Column(db.Integer, default=0)
    
    # Messages et logs
    log_messages = db.Column(db.Text)  # JSON array of log messages
    error_message = db.Column(db.Text)
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Relations
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False, index=True)
    
    def get_duration(self):
        """Retourne la durée de la tâche."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def get_log_messages(self):
        """Retourne les messages de log."""
        if self.log_messages:
            return json.loads(self.log_messages)
        return []
    
    def add_log_message(self, message, level='info'):
        """Ajoute un message de log."""
        logs = self.get_log_messages()
        logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        })
        self.log_messages = json.dumps(logs, ensure_ascii=False)
    
    def to_dict(self):
        """Convertit la tâche en dictionnaire."""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress,
            'cleaning_mode': self.cleaning_mode,
            'original_rows': self.original_rows,
            'cleaned_rows': self.cleaned_rows,
            'removed_duplicates': self.removed_duplicates,
            'filled_missing_values': self.filled_missing_values,
            'detected_outliers': self.detected_outliers,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration': str(self.get_duration()) if self.get_duration() else None,
            'dataset_name': self.dataset.name if self.dataset else None
        }

class AnalysisResult(db.Model):
    """Modèle pour les résultats d'analyse."""
    
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # statistical, ml, clustering, correlation
    
    # Résultats (JSON)
    results_data = db.Column(db.Text)  # JSON string with analysis results
    visualizations = db.Column(db.Text)  # JSON string with chart configurations
    insights = db.Column(db.Text)  # JSON array of insights and recommendations
    
    # Métadonnées
    execution_time = db.Column(db.Float)  # En secondes
    parameters_used = db.Column(db.Text)  # JSON string with analysis parameters
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relations
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False, index=True)
    
    def get_results_data(self):
        """Retourne les données de résultat."""
        if self.results_data:
            return json.loads(self.results_data)
        return {}
    
    def set_results_data(self, data):
        """Définit les données de résultat."""
        self.results_data = json.dumps(data, ensure_ascii=False, default=str)
    
    def get_visualizations(self):
        """Retourne les configurations de visualisation."""
        if self.visualizations:
            return json.loads(self.visualizations)
        return {}
    
    def set_visualizations(self, data):
        """Définit les configurations de visualisation."""
        self.visualizations = json.dumps(data, ensure_ascii=False)
    
    def get_insights(self):
        """Retourne les insights."""
        if self.insights:
            return json.loads(self.insights)
        return []
    
    def set_insights(self, data):
        """Définit les insights."""
        self.insights = json.dumps(data, ensure_ascii=False)
    
    def to_dict(self):
        """Convertit l'analyse en dictionnaire."""
        return {
            'id': self.id,
            'name': self.name,
            'analysis_type': self.analysis_type,
            'results_data': self.get_results_data(),
            'visualizations': self.get_visualizations(),
            'insights': self.get_insights(),
            'execution_time': self.execution_time,
            'created_at': self.created_at.isoformat(),
            'dataset_name': self.dataset.name if self.dataset else None
        }

class APIToken(db.Model):
    """Modèle pour les tokens API."""
    
    __tablename__ = 'api_tokens'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    token_hash = db.Column(db.String(200), nullable=False, unique=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Permissions
    can_read = db.Column(db.Boolean, default=True, nullable=False)
    can_write = db.Column(db.Boolean, default=False, nullable=False)
    can_delete = db.Column(db.Boolean, default=False, nullable=False)
    
    # Statistiques d'usage
    usage_count = db.Column(db.Integer, default=0, nullable=False)
    last_used = db.Column(db.DateTime)
    
    # Dates
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime)
    
    # Relations
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    @classmethod
    def generate_token(cls):
        """Génère un token API unique."""
        return secrets.token_urlsafe(32)
    
    def set_token(self, token):
        """Hash et définit le token."""
        self.token_hash = generate_password_hash(token)
    
    def check_token(self, token):
        """Vérifie le token."""
        return check_password_hash(self.token_hash, token)
    
    def is_expired(self):
        """Vérifie si le token est expiré."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def record_usage(self):
        """Enregistre l'utilisation du token."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        db.session.commit()

# Formulaires WTF
class LoginForm(FlaskForm):
    """Formulaire de connexion."""
    
    email = StringField('Email', validators=[
        DataRequired(message='L\'email est requis'),
        Email(message='Email invalide')
    ], render_kw={'placeholder': 'votre@email.com', 'class': 'form-control'})
    
    password = PasswordField('Mot de passe', validators=[
        DataRequired(message='Le mot de passe est requis'),
        Length(min=6, message='Le mot de passe doit contenir au moins 6 caractères')
    ], render_kw={'placeholder': '••••••••', 'class': 'form-control'})
    
    remember_me = BooleanField('Se souvenir de moi', render_kw={'class': 'form-check-input'})
    
    submit = SubmitField('Se connecter', render_kw={'class': 'btn btn-primary btn-lg w-100'})

class RegisterForm(FlaskForm):
    """Formulaire d'inscription."""
    
    username = StringField('Nom d\'utilisateur', validators=[
        DataRequired(message='Le nom d\'utilisateur est requis'),
        Length(min=3, max=80, message='Le nom d\'utilisateur doit contenir entre 3 et 80 caractères')
    ], render_kw={'placeholder': 'nom_utilisateur', 'class': 'form-control'})
    
    first_name = StringField('Prénom', validators=[
        DataRequired(message='Le prénom est requis'),
        Length(max=50, message='Le prénom ne peut pas dépasser 50 caractères')
    ], render_kw={'placeholder': 'Jean', 'class': 'form-control'})
    
    last_name = StringField('Nom', validators=[
        DataRequired(message='Le nom est requis'),
        Length(max=50, message='Le nom ne peut pas dépasser 50 caractères')
    ], render_kw={'placeholder': 'Dupont', 'class': 'form-control'})
    
    email = StringField('Email', validators=[
        DataRequired(message='L\'email est requis'),
        Email(message='Email invalide'),
        Length(max=120, message='L\'email ne peut pas dépasser 120 caractères')
    ], render_kw={'placeholder': 'jean.dupont@email.com', 'class': 'form-control'})
    
    password = PasswordField('Mot de passe', validators=[
        DataRequired(message='Le mot de passe est requis'),
        Length(min=8, message='Le mot de passe doit contenir au moins 8 caractères')
    ], render_kw={'placeholder': '••••••••', 'class': 'form-control'})
    
    password2 = PasswordField('Confirmer le mot de passe', validators=[
        DataRequired(message='La confirmation du mot de passe est requise'),
        EqualTo('password', message='Les mots de passe ne correspondent pas')
    ], render_kw={'placeholder': '••••••••', 'class': 'form-control'})
    
    accept_terms = BooleanField('J\'accepte les conditions d\'utilisation', validators=[
        DataRequired(message='Vous devez accepter les conditions d\'utilisation')
    ], render_kw={'class': 'form-check-input'})
    
    submit = SubmitField('Créer un compte', render_kw={'class': 'btn btn-primary btn-lg w-100'})
    
    def validate_username(self, username):
        """Valide l'unicité du nom d'utilisateur."""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Ce nom d\'utilisateur est déjà pris.')
    
    def validate_email(self, email):
        """Valide l'unicité de l'email."""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Cet email est déjà enregistré.')

class UploadForm(FlaskForm):
    """Formulaire d'upload de dataset."""
    
    dataset_name = StringField('Nom du dataset', validators=[
        DataRequired(message='Le nom du dataset est requis'),
        Length(min=3, max=200, message='Le nom doit contenir entre 3 et 200 caractères')
    ], render_kw={'placeholder': 'Mon dataset', 'class': 'form-control'})
    
    file = FileField('Fichier de données', validators=[
        FileRequired(message='Veuillez sélectionner un fichier'),
        FileAllowed(['csv', 'xlsx', 'xls', 'json', 'tsv'], 
                   message='Format de fichier non supporté. Utilisez CSV, Excel, JSON ou TSV.')
    ], render_kw={'class': 'form-control', 'accept': '.csv,.xlsx,.xls,.json,.tsv'})
    
    description = TextAreaField('Description (optionnel)', validators=[
        Optional(),
        Length(max=1000, message='La description ne peut pas dépasser 1000 caractères')
    ], render_kw={'placeholder': 'Décrivez votre dataset...', 'class': 'form-control', 'rows': '3'})
    
    submit = SubmitField('Uploader', render_kw={'class': 'btn btn-primary btn-lg'})

class CleaningForm(FlaskForm):
    """Formulaire de configuration du nettoyage."""
    
    job_name = StringField('Nom de la tâche', validators=[
        DataRequired(message='Le nom de la tâche est requis'),
        Length(min=3, max=200, message='Le nom doit contenir entre 3 et 200 caractères')
    ], render_kw={'class': 'form-control'})
    
    cleaning_mode = SelectField('Mode de nettoyage', choices=[
        ('automatic', 'Automatique (recommandé)'),
        ('conservative', 'Conservateur'),
        ('aggressive', 'Agressif'),
        ('custom', 'Personnalisé')
    ], validators=[DataRequired()], render_kw={'class': 'form-select'})
    
    remove_duplicates = BooleanField('Supprimer les doublons', 
                                   default=True, render_kw={'class': 'form-check-input'})
    
    handle_missing_values = SelectField('Traitement des valeurs manquantes', choices=[
        ('auto', 'Automatique'),
        ('drop', 'Supprimer les lignes'),
        ('fill_mean', 'Remplacer par la moyenne'),
        ('fill_median', 'Remplacer par la médiane'),
        ('fill_mode', 'Remplacer par la valeur la plus fréquente'),
        ('interpolate', 'Interpolation'),
        ('forward_fill', 'Report vers l\'avant'),
        ('backward_fill', 'Report vers l\'arrière')
    ], validators=[DataRequired()], render_kw={'class': 'form-select'})
    
    detect_outliers = BooleanField('Détecter les valeurs aberrantes', 
                                 default=True, render_kw={'class': 'form-check-input'})
    
    outlier_method = SelectField('Méthode de détection des outliers', choices=[
        ('iqr', 'Méthode IQR (Inter-Quartile Range)'),
        ('zscore', 'Z-Score'),
        ('isolation_forest', 'Isolation Forest'),
        ('local_outlier_factor', 'Local Outlier Factor')
    ], render_kw={'class': 'form-select'})
    
    standardize_formats = BooleanField('Standardiser les formats', 
                                     default=True, render_kw={'class': 'form-check-input'})
    
    normalize_text = BooleanField('Normaliser le texte (casse, espaces)', 
                                default=True, render_kw={'class': 'form-check-input'})
    
    convert_data_types = BooleanField('Optimiser les types de données', 
                                    default=True, render_kw={'class': 'form-check-input'})
    
    backup_original = BooleanField('Sauvegarder l\'original', 
                                 default=True, render_kw={'class': 'form-check-input'})
    
    submit = SubmitField('Lancer le nettoyage', render_kw={'class': 'btn btn-success btn-lg'})

class AnalysisForm(FlaskForm):
    """Formulaire de configuration de l'analyse."""
    
    analysis_name = StringField('Nom de l\'analyse', validators=[
        DataRequired(message='Le nom de l\'analyse est requis'),
        Length(min=3, max=200, message='Le nom doit contenir entre 3 et 200 caractères')
    ], render_kw={'class': 'form-control'})
    
    analysis_types = SelectField('Type d\'analyse', choices=[
        ('statistical', 'Analyse statistique complète'),
        ('correlation', 'Analyse de corrélation'),
        ('distribution', 'Analyse de distribution'),
        ('clustering', 'Clustering automatique'),
        ('classification', 'Classification automatique'),
        ('regression', 'Régression automatique'),
        ('time_series', 'Analyse de séries temporelles'),
        ('text_analysis', 'Analyse de texte'),
        ('custom', 'Analyse personnalisée')
    ], validators=[DataRequired()], render_kw={'class': 'form-select'})
    
    target_column = SelectField('Colonne cible (optionnel)', choices=[],
                              render_kw={'class': 'form-select'})
    
    feature_columns = SelectField('Colonnes de caractéristiques', choices=[],
                                render_kw={'class': 'form-select', 'multiple': True})
    
    include_visualizations = BooleanField('Générer des visualisations', 
                                        default=True, render_kw={'class': 'form-check-input'})
    
    generate_insights = BooleanField('Générer des insights IA', 
                                   default=True, render_kw={'class': 'form-check-input'})
    
    advanced_ml = BooleanField('Machine Learning avancé', 
                             default=False, render_kw={'class': 'form-check-input'})
    
    export_results = BooleanField('Exporter les résultats', 
                                default=True, render_kw={'class': 'form-check-input'})
    
    submit = SubmitField('Lancer l\'analyse', render_kw={'class': 'btn btn-info btn-lg'})

class ProfileForm(FlaskForm):
    """Formulaire de modification du profil."""
    
    first_name = StringField('Prénom', validators=[
        DataRequired(message='Le prénom est requis'),
        Length(max=50, message='Le prénom ne peut pas dépasser 50 caractères')
    ], render_kw={'class': 'form-control'})
    
    last_name = StringField('Nom', validators=[
        DataRequired(message='Le nom est requis'),
        Length(max=50, message='Le nom ne peut pas dépasser 50 caractères')
    ], render_kw={'class': 'form-control'})
    
    email = StringField('Email', validators=[
        DataRequired(message='L\'email est requis'),
        Email(message='Email invalide')
    ], render_kw={'class': 'form-control'})
    
    default_cleaning_mode = SelectField('Mode de nettoyage par défaut', choices=[
        ('automatic', 'Automatique'),
        ('conservative', 'Conservateur'),
        ('aggressive', 'Agressif')
    ], render_kw={'class': 'form-select'})
    
    auto_analysis_enabled = BooleanField('Analyse automatique après nettoyage', 
                                       render_kw={'class': 'form-check-input'})
    
    submit = SubmitField('Mettre à jour', render_kw={'class': 'btn btn-primary'})
    
    def __init__(self, original_email, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_email = original_email
    
    def validate_email(self, email):
        """Valide l'unicité de l'email."""
        if email.data != self.original_email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('Cet email est déjà utilisé.')

class PasswordChangeForm(FlaskForm):
    """Formulaire de changement de mot de passe."""
    
    current_password = PasswordField('Mot de passe actuel', validators=[
        DataRequired(message='Le mot de passe actuel est requis')
    ], render_kw={'class': 'form-control'})
    
    new_password = PasswordField('Nouveau mot de passe', validators=[
        DataRequired(message='Le nouveau mot de passe est requis'),
        Length(min=8, message='Le mot de passe doit contenir au moins 8 caractères')
    ], render_kw={'class': 'form-control'})
    
    confirm_password = PasswordField('Confirmer le nouveau mot de passe', validators=[
        DataRequired(message='La confirmation est requise'),
        EqualTo('new_password', message='Les mots de passe ne correspondent pas')
    ], render_kw={'class': 'form-control'})
    
    submit = SubmitField('Changer le mot de passe', render_kw={'class': 'btn btn-warning'})

# Services et utilitaires
class FileManager:
    """Gestionnaire de fichiers avancé."""
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.allowed_extensions = current_app.config['ALLOWED_EXTENSIONS']
    
    @staticmethod
    def allowed_file(filename):
        """Vérifie si le fichier est autorisé."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']
    
    @staticmethod
    def get_file_size(file_path):
        """Retourne la taille du fichier en bytes."""
        return os.path.getsize(file_path)
    
    @staticmethod
    def format_file_size(size_bytes):
        """Formate la taille du fichier."""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def save_file(self, file, user_id):
        """Sauvegarde un fichier uploadé."""
        if not file or not self.allowed_file(file.filename):
            return None
        
        # Générer un nom de fichier unique
        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{user_id}_{timestamp}_{filename}"
        
        # Créer le dossier utilisateur si nécessaire
        user_folder = os.path.join(self.upload_folder, str(user_id))
        os.makedirs(user_folder, exist_ok=True)
        
        # Sauvegarder le fichier
        file_path = os.path.join(user_folder, unique_filename)
        file.save(file_path)
        
        return {
            'filename': filename,
            'unique_filename': unique_filename,
            'file_path': file_path,
            'file_size': self.get_file_size(file_path)
        }
    
    def delete_file(self, file_path):
        """Supprime un fichier."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la suppression du fichier {file_path}: {e}")
        return False

class DataProcessor:
    """Processeur de données avancé avec IA."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json', 'tsv']
        self.encoding_options = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def load_dataset(self, file_path, file_type, encoding='utf-8'):
        """Charge un dataset depuis un fichier."""
        try:
            if file_type == 'csv':
                return self._load_csv(file_path, encoding)
            elif file_type in ['xlsx', 'xls']:
                return self._load_excel(file_path)
            elif file_type == 'json':
                return self._load_json(file_path, encoding)
            elif file_type == 'tsv':
                return self._load_tsv(file_path, encoding)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_type}")
        except Exception as e:
            current_app.logger.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path, encoding):
        """Charge un fichier CSV avec détection intelligente."""
        # Tentative avec l'encodage spécifié
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            # Essayer d'autres encodages
            for enc in self.encoding_options:
                if enc != encoding:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        current_app.logger.info(f"Fichier chargé avec l'encodage {enc}")
                        return df
                    except UnicodeDecodeError:
                        continue
            raise ValueError("Impossible de détecter l'encodage du fichier")
    
    def _load_excel(self, file_path):
        """Charge un fichier Excel."""
        try:
            # Lire la première feuille par défaut
            df = pd.read_excel(file_path, engine='openpyxl')
            return df
        except Exception as e:
            # Essayer avec xlrd pour les anciens formats
            try:
                df = pd.read_excel(file_path, engine='xlrd')
                return df
            except:
                raise e
    
    def _load_json(self, file_path, encoding):
        """Charge un fichier JSON."""
        try:
            df = pd.read_json(file_path, encoding=encoding)
            return df
        except Exception as e:
            # Essayer de charger comme JSON lines
            try:
                df = pd.read_json(file_path, lines=True, encoding=encoding)
                return df
            except:
                raise e
    
    def _load_tsv(self, file_path, encoding):
        """Charge un fichier TSV."""
        try:
            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
            return df
        except UnicodeDecodeError:
            # Essayer d'autres encodages
            for enc in self.encoding_options:
                if enc != encoding:
                    try:
                        df = pd.read_csv(file_path, sep='\t', encoding=enc)
                        return df
                    except UnicodeDecodeError:
                        continue
            raise ValueError("Impossible de détecter l'encodage du fichier")
    
    def analyze_dataset_quality(self, df):
        """Analyse la qualité d'un dataset."""
        analysis = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'data_types': df.dtypes.to_dict()
            },
            'missing_values': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum().to_dict(),
                'percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'duplicates': {
                'count': df.duplicated().sum(),
                'percentage': df.duplicated().sum() / len(df) * 100
            },
            'data_quality_score': 0
        }
        
        # Calcul du score de qualité (0-100)
        score = 100
        
        # Pénalité pour les valeurs manquantes
        missing_penalty = min(analysis['missing_values']['total'] / (len(df) * len(df.columns)) * 50, 30)
        score -= missing_penalty
        
        # Pénalité pour les doublons
        duplicate_penalty = min(analysis['duplicates']['percentage'] / 2, 20)
        score -= duplicate_penalty
        
        # Bonus pour la diversité des types de données
        unique_types = len(set(str(dtype) for dtype in df.dtypes))
        type_bonus = min(unique_types * 2, 10)
        score += type_bonus
        
        analysis['data_quality_score'] = max(0, min(100, score))
        
        return analysis
    
    def detect_column_types(self, df):
        """Détecte intelligemment les types de colonnes."""
        column_types = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                column_types[col] = 'empty'
                continue
            
            # Détection des dates
            if self._is_datetime_column(series):
                column_types[col] = 'datetime'
            # Détection des nombres
            elif pd.api.types.is_numeric_dtype(series):
                if series.dtype == 'int64' or all(series == series.astype(int)):
                    column_types[col] = 'integer'
                else:
                    column_types[col] = 'float'
            # Détection des booléens
            elif self._is_boolean_column(series):
                column_types[col] = 'boolean'
            # Détection des catégories
            elif self._is_categorical_column(series):
                column_types[col] = 'categorical'
            # Détection des emails
            elif self._is_email_column(series):
                column_types[col] = 'email'
            # Détection des URLs
            elif self._is_url_column(series):
                column_types[col] = 'url'
            # Détection des IDs
            elif self._is_id_column(col, series):
                column_types[col] = 'identifier'
            # Par défaut: texte
            else:
                column_types[col] = 'text'
        
        return column_types
    
    def _is_datetime_column(self, series):
        """Détecte si une colonne contient des dates."""
        if series.dtype == 'datetime64[ns]':
            return True
        
        # Essayer de parser quelques valeurs comme des dates
        sample = series.head(10).astype(str)
        date_count = 0
        
        for value in sample:
            try:
                pd.to_datetime(value)
                date_count += 1
            except:
                continue
        
        return date_count / len(sample) > 0.5
    
    def _is_boolean_column(self, series):
        """Détecte si une colonne contient des valeurs booléennes."""
        if series.dtype == 'bool':
            return True
        
        unique_values = set(series.astype(str).str.lower().unique())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'oui', 'non'}
        
        return unique_values.issubset(boolean_values) and len(unique_values) <= 4
    
    def _is_categorical_column(self, series):
        """Détecte si une colonne est catégorielle."""
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.05 or (unique_ratio < 0.1 and series.nunique() < 20)
    
    def _is_email_column(self, series):
        """Détecte si une colonne contient des emails."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        sample = series.head(10).astype(str)
        email_count = sum(1 for value in sample if email_pattern.match(value))
        
        return email_count / len(sample) > 0.7
    
    def _is_url_column(self, series):
        """Détecte si une colonne contient des URLs."""
        import re
        url_pattern = re.compile(r'^https?://|^www\.|\.com$|\.org$|\.net$')
        
        sample = series.head(10).astype(str)
        url_count = sum(1 for value in sample if url_pattern.search(value))
        
        return url_count / len(sample) > 0.7
    
    def _is_id_column(self, column_name, series):
        """Détecte si une colonne est un identifiant."""
        name_indicators = ['id', 'identifier', 'key', 'pk', 'primary']
        
        # Vérifier le nom de la colonne
        col_lower = column_name.lower()
        if any(indicator in col_lower for indicator in name_indicators):
            return True
        
        # Vérifier l'unicité
        if series.nunique() == len(series):
            return True
        
        return False

class DataCleaner:
    """Nettoyeur de données intelligent avec IA."""
    
    def __init__(self):
        self.processor = DataProcessor()
    
    def clean_dataset(self, df, cleaning_config, job=None):
        """Nettoie un dataset selon la configuration."""
        try:
            original_shape = df.shape
            cleaning_report = {
                'original_rows': original_shape[0],
                'original_columns': original_shape[1],
                'operations': [],
                'warnings': [],
                'errors': []
            }
            
            if job:
                job.add_log_message("Début du nettoyage des données", "info")
                job.progress = 10
                db.session.commit()
            
            # 1. Supprimer les doublons
            if cleaning_config.get('remove_duplicates', True):
                duplicates_count = df.duplicated().sum()
                if duplicates_count > 0:
                    df = df.drop_duplicates()
                    cleaning_report['operations'].append({
                        'operation': 'remove_duplicates',
                        'count': duplicates_count,
                        'description': f"Suppression de {duplicates_count} doublons"
                    })
                    if job:
                        job.add_log_message(f"Suppression de {duplicates_count} doublons", "info")
            
            if job:
                job.progress = 25
                db.session.commit()
            
            # 2. Traitement des valeurs manquantes
            missing_method = cleaning_config.get('handle_missing_values', 'auto')
            df, missing_report = self._handle_missing_values(df, missing_method)
            cleaning_report['operations'].extend(missing_report)
            
            if job:
                job.progress = 50
                db.session.commit()
            
            # 3. Détection et traitement des outliers
            if cleaning_config.get('detect_outliers', True):
                outlier_method = cleaning_config.get('outlier_method', 'iqr')
                df, outlier_report = self._detect_and_handle_outliers(df, outlier_method)
                cleaning_report['operations'].extend(outlier_report)
                
                if job:
                    job.add_log_message(f"Détection des outliers avec méthode {outlier_method}", "info")
            
            if job:
                job.progress = 70
                db.session.commit()
            
            # 4. Standardisation des formats
            if cleaning_config.get('standardize_formats', True):
                df, format_report = self._standardize_formats(df)
                cleaning_report['operations'].extend(format_report)
            
            # 5. Normalisation du texte
            if cleaning_config.get('normalize_text', True):
                df, text_report = self._normalize_text(df)
                cleaning_report['operations'].extend(text_report)
            
            if job:
                job.progress = 85
                db.session.commit()
            
            # 6. Optimisation des types de données
            if cleaning_config.get('convert_data_types', True):
                df, type_report = self._optimize_data_types(df)
                cleaning_report['operations'].extend(type_report)
            
            # 7. Validation finale
            df = self._validate_data(df)
            
            if job:
                job.progress = 100
                job.add_log_message("Nettoyage terminé avec succès", "success")
                db.session.commit()
            
            cleaning_report['final_rows'] = df.shape[0]
            cleaning_report['final_columns'] = df.shape[1]
            cleaning_report['rows_removed'] = original_shape[0] - df.shape[0]
            cleaning_report['data_reduction'] = (1 - df.shape[0] / original_shape[0]) * 100
            
            return df, cleaning_report
            
        except Exception as e:
            if job:
                job.add_log_message(f"Erreur lors du nettoyage: {str(e)}", "error")
                job.error_message = str(e)
                db.session.commit()
            raise
    
    def _handle_missing_values(self, df, method):
        """Traite les valeurs manquantes."""
        report = []
        original_missing = df.isnull().sum().sum()
        
        if original_missing == 0:
            return df, report
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            column_method = method
            if method == 'auto':
                # Choix automatique basé sur le type de données
                if pd.api.types.is_numeric_dtype(df[column]):
                    column_method = 'fill_median'
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    column_method = 'interpolate'
                else:
                    column_method = 'fill_mode'
            
            if column_method == 'drop':
                df = df.dropna(subset=[column])
            elif column_method == 'fill_mean' and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            elif column_method == 'fill_median' and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            elif column_method == 'fill_mode':
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value[0])
            elif column_method == 'interpolate' and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].interpolate()
            elif column_method == 'forward_fill':
                df[column] = df[column].fillna(method='ffill')
            elif column_method == 'backward_fill':
                df[column] = df[column].fillna(method='bfill')
            
            report.append({
                'operation': f'missing_values_{column_method}',
                'column': column,
                'count': missing_count,
                'description': f"Traitement de {missing_count} valeurs manquantes dans {column}"
            })
        
        return df, report
    
    def _detect_and_handle_outliers(self, df, method):
        """Détecte et traite les outliers."""
        report = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].nunique() < 3:  # Skip columns with very few unique values
                continue
            
            outliers_mask = self._detect_outliers(df[column], method)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                # Pour l'instant, on marque seulement les outliers sans les supprimer
                # Dans une version plus avancée, on pourrait les traiter différemment
                report.append({
                    'operation': f'outliers_detected_{method}',
                    'column': column,
                    'count': outliers_count,
                    'description': f"Détection de {outliers_count} outliers dans {column}"
                })
        
        return df, report
    
    def _detect_outliers(self, series, method):
        """Détecte les outliers selon différentes méthodes."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = 3
            return pd.Series(z_scores > threshold, index=series.index).fillna(False)
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            outliers = clf.fit_predict(series.values.reshape(-1, 1))
            return pd.Series(outliers == -1, index=series.index)
        
        elif method == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            clf = LocalOutlierFactor(contamination=0.1)
            outliers = clf.fit_predict(series.values.reshape(-1, 1))
            return pd.Series(outliers == -1, index=series.index)
        
        return pd.Series(False, index=series.index)
    
    def _standardize_formats(self, df):
        """Standardise les formats de données."""
        report = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Standardiser les dates
                if self.processor._is_datetime_column(df[column]):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        report.append({
                            'operation': 'standardize_datetime',
                            'column': column,
                            'description': f"Standardisation des dates dans {column}"
                        })
                    except:
                        pass
                
                # Standardiser les booléens
                elif self.processor._is_boolean_column(df[column]):
                    bool_map = {
                        'true': True, 'false': False, '1': True, '0': False,
                        'yes': True, 'no': False, 'y': True, 'n': False,
                        'oui': True, 'non': False
                    }
                    df[column] = df[column].astype(str).str.lower().map(bool_map)
                    report.append({
                        'operation': 'standardize_boolean',
                        'column': column,
                        'description': f"Standardisation des valeurs booléennes dans {column}"
                    })
        
        return df, report
    
    def _normalize_text(self, df):
        """Normalise le texte."""
        report = []
        text_columns = df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            if df[column].dtype == 'object':
                # Supprimer les espaces en début/fin
                df[column] = df[column].astype(str).str.strip()
                
                # Remplacer les espaces multiples par un seul
                df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
                
                report.append({
                    'operation': 'normalize_text',
                    'column': column,
                    'description': f"Normalisation du texte dans {column}"
                })
        
        return df, report
    
    def _optimize_data_types(self, df):
        """Optimise les types de données pour réduire l'utilisation mémoire."""
        report = []
        
        for column in df.columns:
            original_dtype = df[column].dtype
            optimized = False
            
            # Optimiser les entiers
            if pd.api.types.is_integer_dtype(df[column]):
                c_min = df[column].min()
                c_max = df[column].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                    optimized = True
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                    optimized = True
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                    optimized = True
            
            # Optimiser les flottants
            elif pd.api.types.is_float_dtype(df[column]):
                c_min = df[column].min()
                c_max = df[column].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[column] = df[column].astype(np.float32)
                    optimized = True
            
            # Convertir en catégoriel si approprié
            elif df[column].dtype == 'object':
                unique_ratio = df[column].nunique() / len(df[column])
                if unique_ratio < 0.5:  # Si moins de 50% de valeurs uniques
                    df[column] = df[column].astype('category')
                    optimized = True
            
            if optimized:
                report.append({
                    'operation': 'optimize_dtype',
                    'column': column,
                    'original_dtype': str(original_dtype),
                    'new_dtype': str(df[column].dtype),
                    'description': f"Optimisation du type de {column}: {original_dtype} → {df[column].dtype}"
                })
        
        return df, report
    
    def _validate_data(self, df):
        """Validation finale des données."""
        # Supprimer les colonnes entièrement vides
        df = df.dropna(axis=1, how='all')
        
        # Supprimer les lignes entièrement vides
        df = df.dropna(axis=0, how='all')
        
        # Réinitialiser l'index
        df = df.reset_index(drop=True)
        
        return df

class MLAnalyzer:
    """Analyseur de Machine Learning automatique."""
    
    def __init__(self):
        self.feature_importance_threshold = 0.01
        self.correlation_threshold = 0.8
    
    def auto_analyze(self, df, target_column=None, analysis_type='statistical'):
        """Effectue une analyse automatique."""
        try:
            results = {
                'analysis_type': analysis_type,
                'timestamp': datetime.utcnow().isoformat(),
                'dataset_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
            }
            
            if analysis_type == 'statistical':
                results.update(self._statistical_analysis(df))
            elif analysis_type == 'correlation':
                results.update(self._correlation_analysis(df))
            elif analysis_type == 'clustering':
                results.update(self._clustering_analysis(df))
            elif analysis_type == 'classification' and target_column:
                results.update(self._classification_analysis(df, target_column))
            elif analysis_type == 'regression' and target_column:
                results.update(self._regression_analysis(df, target_column))
            
            return results
            
        except Exception as e:
            current_app.logger.error(f"Erreur lors de l'analyse ML: {e}")
            raise
    
    def _statistical_analysis(self, df):
        """Analyse statistique complète."""
        results = {}
        
        # Statistiques descriptives
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            results['descriptive_stats'] = {
                'summary': numeric_df.describe().to_dict(),
                'skewness': numeric_df.skew().to_dict(),
                'kurtosis': numeric_df.kurtosis().to_dict()
            }
        
        # Analyse des valeurs manquantes
        results['missing_analysis'] = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Analyse de la distribution
        results['distribution_analysis'] = {}
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) > 0:
                results['distribution_analysis'][column] = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'quartiles': {
                        'q1': float(series.quantile(0.25)),
                        'q3': float(series.quantile(0.75))
                    },
                    'is_normal_distributed': self._test_normality(series)
                }
        
        # Analyse des colonnes catégorielles
        categorical_df = df.select_dtypes(include=['object', 'category'])
        results['categorical_analysis'] = {}
        for column in categorical_df.columns:
            series = categorical_df[column].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                results['categorical_analysis'][column] = {
                    'unique_count': int(series.nunique()),
                    'most_frequent': str(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0]),
                    'frequency_distribution': value_counts.head(10).to_dict()
                }
        
        return results
    
    def _correlation_analysis(self, df):
        """Analyse de corrélation."""
        results = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            results['error'] = "Pas assez de colonnes numériques pour l'analyse de corrélation"
            return results
        
        # Matrice de corrélation
        correlation_matrix = numeric_df.corr()
        results['correlation_matrix'] = correlation_matrix.to_dict()
        
        # Paires fortement corrélées
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.correlation_threshold:
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.9 else 'moderate'
                    })
        
        results['strong_correlations'] = strong_correlations
        
        # Suggestions basées sur les corrélations
        results['suggestions'] = []
        if len(strong_correlations) > 0:
            results['suggestions'].append({
                'type': 'correlation',
                'message': f"Trouvé {len(strong_correlations)} corrélations fortes. Considérez la réduction de dimensionnalité.",
                'priority': 'medium'
            })
        
        return results
    
    def _clustering_analysis(self, df):
        """Analyse de clustering automatique."""
        results = {}
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
            results['error'] = "Données insuffisantes pour le clustering"
            return results
        
        # Préparation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Détermination du nombre optimal de clusters (méthode du coude)
        max_clusters = min(10, numeric_df.shape[0] // 2)
        inertias = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Trouver le coude (différence de pente maximale)
        optimal_k = 3  # Valeur par défaut
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                optimal_k = np.argmax(second_diffs) + 2
        
        # Clustering final
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Analyse des clusters
        results['clustering_results'] = {
            'optimal_clusters': int(optimal_k),
            'cluster_sizes': {f'cluster_{i}': int(np.sum(cluster_labels == i)) 
                            for i in range(optimal_k)},
            'inertia': float(kmeans.inertia_),
            'silhouette_score': self._calculate_silhouette_score(scaled_data, cluster_labels)
        }
        
        # Caractéristiques des clusters
        df_with_clusters = numeric_df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        cluster_profiles = {}
        for i in range(optimal_k):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            cluster_profiles[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'centroid': cluster_data.drop('cluster', axis=1).mean().to_dict()
            }
        
        results['cluster_profiles'] = cluster_profiles
        
        return results
    
    def _classification_analysis(self, df, target_column):
        """Analyse de classification automatique."""
        results = {}
        
        if target_column not in df.columns:
            results['error'] = f"Colonne cible '{target_column}' non trouvée"
            return results
        
        # Préparation des données
        y = df[target_column].dropna()
        X = df.drop(target_column, axis=1)
        
        # Garder seulement les lignes où y n'est pas nul
        valid_indices = y.index
        X = X.loc[valid_indices]
        
        # Sélection des features numériques
        numeric_features = X.select_dtypes(include=[np.number])
        if numeric_features.empty:
            results['error'] = "Aucune feature numérique disponible"
            return results
        
        # Encodage de la variable cible si nécessaire
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y.astype(str))
            class_names = label_encoder.classes_.tolist()
        else:
            y_encoded = y.values
            class_names = sorted(y.unique().tolist())
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Entraînement du modèle
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions et métriques
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        
        # Importance des features
        feature_importance = dict(zip(numeric_features.columns, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        results['classification_results'] = {
            'accuracy': float(accuracy),
            'n_classes': len(class_names),
            'class_names': class_names,
            'feature_importance': dict(sorted_features),
            'top_features': [feat[0] for feat in sorted_features[:5]],
            'model_type': 'RandomForest'
        }
        
        return results
    
    def _regression_analysis(self, df, target_column):
        """Analyse de régression automatique."""
        results = {}
        
        if target_column not in df.columns:
            results['error'] = f"Colonne cible '{target_column}' non trouvée"
            return results
        
        # Vérifier que la cible est numérique
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            results['error'] = f"La colonne cible '{target_column}' doit être numérique pour la régression"
            return results
        
        # Préparation des données
        y = df[target_column].dropna()
        X = df.drop(target_column, axis=1)
        
        # Garder seulement les lignes où y n'est pas nul
        valid_indices = y.index
        X = X.loc[valid_indices]
        
        # Sélection des features numériques
        numeric_features = X.select_dtypes(include=[np.number])
        if numeric_features.empty:
            results['error'] = "Aucune feature numérique disponible"
            return results
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features, y, test_size=0.2, random_state=42
        )
        
        # Entraînement du modèle
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions et métriques
        y_pred = model.predict(X_test)
        r2_score = model.score(X_test, y_test)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Importance des features
        feature_importance = dict(zip(numeric_features.columns, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        results['regression_results'] = {
            'r2_score': float(r2_score),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'feature_importance': dict(sorted_features),
            'top_features': [feat[0] for feat in sorted_features[:5]],
            'model_type': 'RandomForest'
        }
        
        return results
    
    def _test_normality(self, series):
        """Test de normalité (Shapiro-Wilk pour petits échantillons)."""
        if len(series) < 8:
            return False
        
        try:
            from scipy.stats import shapiro
            # Utiliser un échantillon pour les grandes séries
            sample = series.sample(min(5000, len(series)), random_state=42)
            stat, p_value = shapiro(sample)
            return p_value > 0.05
        except:
            return False
    
    def _calculate_silhouette_score(self, data, labels):
        """Calcule le score de silhouette."""
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                return float(silhouette_score(data, labels))
        except:
            pass
        return 0.0

class VisualizationGenerator:
    """Générateur de visualisations avancées."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    def generate_visualizations(self, df, analysis_results, analysis_type='statistical'):
        """Génère des visualisations basées sur l'analyse."""
        try:
            visualizations = {}
            
            if analysis_type == 'statistical':
                visualizations.update(self._create_statistical_plots(df, analysis_results))
            elif analysis_type == 'correlation':
                visualizations.update(self._create_correlation_plots(df, analysis_results))
            elif analysis_type == 'clustering':
                visualizations.update(self._create_clustering_plots(df, analysis_results))
            
            return visualizations
            
        except Exception as e:
            current_app.logger.error(f"Erreur lors de la génération des visualisations: {e}")
            return {}
    
    def _create_statistical_plots(self, df, results):
        """Crée des graphiques statistiques."""
        plots = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Histogrammes des distributions
        if not numeric_df.empty:
            plots['distributions'] = self._create_distribution_plots(numeric_df)
        
        # Box plots pour détecter les outliers
        if not numeric_df.empty:
            plots['boxplots'] = self._create_boxplots(numeric_df)
        
        # Graphique des valeurs manquantes
        if 'missing_analysis' in results:
            plots['missing_values'] = self._create_missing_values_plot(df, results['missing_analysis'])
        
        return plots
    
    def _create_correlation_plots(self, df, results):
        """Crée des graphiques de corrélation."""
        plots = {}
        
        if 'correlation_matrix' in results:
            plots['correlation_heatmap'] = self._create_correlation_heatmap(results['correlation_matrix'])
        
        return plots
    
    def _create_clustering_plots(self, df, results):
        """Crée des graphiques de clustering."""
        plots = {}
        
        if 'clustering_results' in results:
            plots['cluster_analysis'] = self._create_cluster_plots(df, results)
        
        return plots
    
    def _create_distribution_plots(self, numeric_df):
        """Crée des histogrammes de distribution."""
        n_cols = min(3, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(numeric_df.columns):
            if i < len(axes):
                axes[i].hist(numeric_df[column].dropna(), bins=30, 
                           color=self.color_palette[i % len(self.color_palette)], alpha=0.7)
                axes[i].set_title(f'Distribution de {column}')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Fréquence')
        
        # Masquer les axes non utilisés
        for i in range(len(numeric_df.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'histogram',
            'title': 'Distributions des variables numériques',
            'data': f"data:image/png;base64,{img_base64}"
        }
    
    def _create_boxplots(self, numeric_df):
        """Crée des box plots."""
        n_cols = min(3, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(numeric_df.columns):
            if i < len(axes):
                box_plot = axes[i].boxplot(numeric_df[column].dropna(), patch_artist=True)
                box_plot['boxes'][0].set_facecolor(self.color_palette[i % len(self.color_palette)])
                axes[i].set_title(f'Box Plot de {column}')
                axes[i].set_ylabel(column)
        
        # Masquer les axes non utilisés
        for i in range(len(numeric_df.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'boxplot',
            'title': 'Box Plots - Détection des outliers',
            'data': f"data:image/png;base64,{img_base64}"
        }
    
    def _create_missing_values_plot(self, df, missing_analysis):
        """Crée un graphique des valeurs manquantes."""
        missing_data = pd.Series(missing_analysis['missing_by_column'])
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if missing_data.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(missing_data)), missing_data.values, 
                      color=self.color_palette[0])
        ax.set_yticks(range(len(missing_data)))
        ax.set_yticklabels(missing_data.index)
        ax.set_xlabel('Nombre de valeurs manquantes')
        ax.set_title('Valeurs manquantes par colonne')
        
        # Ajouter les valeurs sur les barres
        for i, (idx, value) in enumerate(missing_data.items()):
            ax.text(value + 0.5, i, str(value), va='center')
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'bar_chart',
            'title': 'Analyse des valeurs manquantes',
            'data': f"data:image/png;base64,{img_base64}"
        }
    
    def _create_correlation_heatmap(self, correlation_matrix):
        """Crée une heatmap de corrélation."""
        correlation_df = pd.DataFrame(correlation_matrix)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        
        sns.heatmap(correlation_df, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('Matrice de corrélation')
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'heatmap',
            'title': 'Matrice de corrélation',
            'data': f"data:image/png;base64,{img_base64}"
        }
    
    def _create_cluster_plots(self, df, results):
        """Crée des visualisations de clustering."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Utiliser les deux premières composantes pour la visualisation
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # PCA pour la visualisation 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # Refaire le clustering sur les données originales
        optimal_k = results['clustering_results']['optimal_clusters']
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot des clusters
        for i in range(optimal_k):
            cluster_data = pca_data[cluster_labels == i]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                      c=self.color_palette[i % len(self.color_palette)], 
                      label=f'Cluster {i}', alpha=0.7, s=50)
        
        # Centres des clusters dans l'espace PCA
        cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                  c='black', marker='x', s=200, linewidths=3, label='Centres')
        
        ax.set_xlabel(f'Première composante (var. expliquée: {pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Deuxième composante (var. expliquée: {pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title(f'Clustering K-Means (k={optimal_k})')
        ax.legend()
        
        plt.tight_layout()
        
        # Convertir en base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            'type': 'scatter',
            'title': f'Analyse de clustering (K-Means, k={optimal_k})',
            'data': f"data:image/png;base64,{img_base64}"
        }

class InsightGenerator:
    """Générateur d'insights IA avancés."""
    
    def __init__(self):
        self.insight_templates = {
            'data_quality': [
                "La qualité des données est {quality_level} avec un score de {score}%.",
                "Attention: {missing_percentage}% des données sont manquantes.",
                "Détection de {duplicate_count} doublons dans le dataset."
            ],
            'distribution': [
                "La colonne '{column}' suit une distribution {distribution_type}.",
                "Détection d'asymétrie importante dans '{column}' (skewness: {skewness}).",
                "La variable '{column}' présente des valeurs aberrantes importantes."
            ],
            'correlation': [
                "Forte corrélation détectée entre '{col1}' et '{col2}' (r={correlation}).",
                "Les variables '{columns}' sont redondantes et peuvent être réduites.",
                "Aucune corrélation significative détectée entre les variables."
            ],
            'ml_performance': [
                "Le modèle {model_type} atteint une précision de {accuracy}%.",
                "Les features les plus importantes sont: {top_features}.",
                "Le clustering révèle {n_clusters} groupes distincts dans les données."
            ]
        }
    
    def generate_insights(self, analysis_results, df):
        """Génère des insights automatiques."""
        insights = []
        
        # Insights sur la qualité des données
        if 'missing_analysis' in analysis_results:
            insights.extend(self._quality_insights(analysis_results, df))
        
        # Insights statistiques
        if 'descriptive_stats' in analysis_results:
            insights.extend(self._statistical_insights(analysis_results))
        
        # Insights de corrélation
        if 'correlation_matrix' in analysis_results:
            insights.extend(self._correlation_insights(analysis_results))
        
        # Insights ML
        if 'classification_results' in analysis_results:
            insights.extend(self._ml_insights(analysis_results, 'classification'))
        elif 'regression_results' in analysis_results:
            insights.extend(self._ml_insights(analysis_results, 'regression'))
        elif 'clustering_results' in analysis_results:
            insights.extend(self._clustering_insights(analysis_results))
        
        return insights
    
    def _quality_insights(self, results, df):
        """Génère des insights sur la qualité des données."""
        insights = []
        missing_analysis = results['missing_analysis']
        
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_analysis['total_missing'] / total_cells) * 100
        
        if missing_percentage < 5:
            quality_level = "excellente"
        elif missing_percentage < 15:
            quality_level = "bonne"
        elif missing_percentage < 30:
            quality_level = "moyenne"
        else:
            quality_level = "faible"
        
        insights.append({
            'type': 'data_quality',
            'priority': 'high' if missing_percentage > 20 else 'medium',
            'title': 'Qualité des données',
            'message': f"La qualité des données est {quality_level} avec {missing_percentage:.1f}% de valeurs manquantes.",
            'actionable': missing_percentage > 10,
            'recommendation': "Considérez un nettoyage approfondi des données." if missing_percentage > 10 else "Qualité satisfaisante pour l'analyse."
        })
        
        # Insights par colonne avec beaucoup de valeurs manquantes
        problematic_columns = [col for col, pct in missing_analysis['missing_percentage'].items() if pct > 25]
        if problematic_columns:
            insights.append({
                'type': 'data_quality',
                'priority': 'high',
                'title': 'Colonnes problématiques',
                'message': f"Les colonnes {', '.join(problematic_columns)} ont plus de 25% de valeurs manquantes.",
                'actionable': True,
                'recommendation': "Évaluez s'il faut supprimer ces colonnes ou imputer les valeurs manquantes."
            })
        
        return insights
    
    def _statistical_insights(self, results):
        """Génère des insights statistiques."""
        insights = []
        
        if 'descriptive_stats' in results:
            stats = results['descriptive_stats']
            
            # Analyse de la variance
            if 'summary' in stats:
                high_variance_cols = []
                for col, data in stats['summary'].items():
                    if 'std' in data and 'mean' in data and data['mean'] != 0:
                        cv = data['std'] / abs(data['mean'])  # Coefficient de variation
                        if cv > 1:  # Variance élevée
                            high_variance_cols.append(col)
                
                if high_variance_cols:
                    insights.append({
                        'type': 'distribution',
                        'priority': 'medium',
                        'title': 'Variance élevée détectée',
                        'message': f"Les colonnes {', '.join(high_variance_cols)} présentent une variance élevée.",
                        'actionable': True,
                        'recommendation': "Considérez une normalisation ou standardisation de ces variables."
                    })
            
            # Analyse de l'asymétrie
            if 'skewness' in stats:
                skewed_cols = {col: skew for col, skew in stats['skewness'].items() if abs(skew) > 1}
                if skewed_cols:
                    insights.append({
                        'type': 'distribution',
                        'priority': 'medium',
                        'title': 'Distributions asymétriques',
                        'message': f"Asymétrie détectée dans {len(skewed_cols)} colonnes.",
                        'actionable': True,
                        'recommendation': "Appliquez une transformation (log, sqrt) pour normaliser ces distributions."
                    })
        
        return insights
    
    def _correlation_insights(self, results):
        """Génère des insights sur les corrélations."""
        insights = []
        
        if 'strong_correlations' in results:
            strong_corr = results['strong_correlations']
            
            if len(strong_corr) > 0:
                very_strong = [corr for corr in strong_corr if abs(corr['correlation']) > 0.9]
                
                insights.append({
                    'type': 'correlation',
                    'priority': 'high' if very_strong else 'medium',
                    'title': 'Corrélations fortes détectées',
                    'message': f"Trouvé {len(strong_corr)} paires de variables fortement corrélées.",
                    'actionable': True,
                    'recommendation': "Considérez la suppression de variables redondantes ou l'application de PCA."
                })
                
                if very_strong:
                    pairs = [f"{corr['column1']}-{corr['column2']}" for corr in very_strong[:3]]
                    insights.append({
                        'type': 'correlation',
                        'priority': 'high',
                        'title': 'Redondance critique',
                        'message': f"Corrélations très fortes (>0.9): {', '.join(pairs)}.",
                        'actionable': True,
                        'recommendation': "Suppression recommandée d'une des variables de chaque paire."
                    })
            else:
                insights.append({
                    'type': 'correlation',
                    'priority': 'low',
                    'title': 'Variables indépendantes',
                    'message': "Aucune corrélation forte détectée entre les variables.",
                    'actionable': False,
                    'recommendation': "Excellente diversité des variables pour l'analyse."
                })
        
        return insights
    
    def _ml_insights(self, results, analysis_type):
        """Génère des insights ML."""
        insights = []
        
        if analysis_type == 'classification' and 'classification_results' in results:
            results_data = results['classification_results']
            accuracy = results_data['accuracy']
            
            if accuracy > 0.9:
                performance_level = "excellente"
                priority = "low"
            elif accuracy > 0.8:
                performance_level = "bonne"
                priority = "medium"
            elif accuracy > 0.7:
                performance_level = "correcte"
                priority = "medium"
            else:
                performance_level = "faible"
                priority = "high"
            
            insights.append({
                'type': 'ml_performance',
                'priority': priority,
                'title': 'Performance de classification',
                'message': f"Précision {performance_level}: {accuracy:.1%}",
                'actionable': accuracy < 0.8,
                'recommendation': "Optimisez les hyperparamètres ou ajoutez plus de features." if accuracy < 0.8 else "Performance satisfaisante."
            })
            
            # Insights sur les features importantes
            top_features = results_data.get('top_features', [])[:3]
            if top_features:
                insights.append({
                    'type': 'feature_importance',
                    'priority': 'medium',
                    'title': 'Variables clés identifiées',
                    'message': f"Les variables les plus importantes sont: {', '.join(top_features)}.",
                    'actionable': True,
                    'recommendation': "Concentrez l'analyse sur ces variables principales."
                })
        
        elif analysis_type == 'regression' and 'regression_results' in results:
            results_data = results['regression_results']
            r2_score = results_data['r2_score']
            
            if r2_score > 0.8:
                performance_level = "excellente"
                priority = "low"
            elif r2_score > 0.6:
                performance_level = "bonne"
                priority = "medium"
            elif r2_score > 0.4:
                performance_level = "correcte"
                priority = "medium"
            else:
                performance_level = "faible"
                priority = "high"
            
            insights.append({
                'type': 'ml_performance',
                'priority': priority,
                'title': 'Performance de régression',
                'message': f"R² {performance_level}: {r2_score:.3f}",
                'actionable': r2_score < 0.6,
                'recommendation': "Ajoutez plus de variables explicatives ou transformez les données." if r2_score < 0.6 else "Modèle explicatif satisfaisant."
            })
        
        return insights
    
    def _clustering_insights(self, results):
        """Génère des insights de clustering."""
        insights = []
        
        if 'clustering_results' in results:
            clustering_data = results['clustering_results']
            n_clusters = clustering_data['optimal_clusters']
            silhouette_score = clustering_data.get('silhouette_score', 0)
            
            if silhouette_score > 0.7:
                cluster_quality = "excellente"
                priority = "low"
            elif silhouette_score > 0.5:
                cluster_quality = "bonne"
                priority = "medium"
            elif silhouette_score > 0.3:
                cluster_quality = "correcte"
                priority = "medium"
            else:
                cluster_quality = "faible"
                priority = "high"
            
            insights.append({
                'type': 'clustering',
                'priority': priority,
                'title': 'Segmentation des données',
                'message': f"Identification de {n_clusters} groupes distincts avec une séparation {cluster_quality}.",
                'actionable': silhouette_score < 0.5,
                'recommendation': "Ajustez le nombre de clusters ou les variables utilisées." if silhouette_score < 0.5 else "Segmentation réussie des données."
            })
            
            # Analyse de la distribution des clusters
            cluster_sizes = clustering_data.get('cluster_sizes', {})
            if cluster_sizes:
                sizes = list(cluster_sizes.values())
                max_size = max(sizes)
                min_size = min(sizes)
                
                if max_size / min_size > 5:  # Déséquilibre important
                    insights.append({
                        'type': 'clustering',
                        'priority': 'medium',
                        'title': 'Déséquilibre des clusters',
                        'message': f"Forte disparité dans la taille des clusters ({min_size} - {max_size} éléments).",
                        'actionable': True,
                        'recommendation': "Vérifiez la présence d'outliers ou ajustez les paramètres de clustering."
                    })
        
        return insights

# Décorateurs et utilitaires
def async_task(f):
    """Décorateur pour les tâches asynchrones."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_app.config.get('ASYNC_PROCESSING_ENABLED', True):
            executor = ThreadPoolExecutor(max_workers=2)
            future = executor.submit(f, *args, **kwargs)
            return future
        else:
            return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Décorateur pour les routes admin seulement."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def premium_required(f):
    """Décorateur pour les fonctionnalités premium."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_premium:
            flash('Cette fonctionnalité nécessite un compte Premium.', 'warning')
            return redirect(url_for('main.pricing'))
        return f(*args, **kwargs)
    return decorated_function

def format_datetime(dt):
    """Formate une datetime pour l'affichage."""
    if dt is None:
        return 'N/A'
    return dt.strftime('%d/%m/%Y à %H:%M')

def format_file_size(size_bytes):
    """Formate une taille de fichier."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def truncate_text(text, length=50):
    """Tronque un texte à une longueur donnée."""
    if len(text) <= length:
        return text
    return text[:length-3] + '...'

# Gestionnaire d'erreurs personnalisés
def create_app(config_name='default'):
    """Factory pour créer l'application Flask."""
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialisation des extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    mail.init_app(app)
    cache.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    
    # Configuration des dossiers
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Configuration du logging
    if not app.debug:
        import logging
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(app.config['LOG_FILE'], maxBytes=10240000, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(app.config['LOG_LEVEL'])
        app.logger.addHandler(file_handler)
        app.logger.setLevel(app.config['LOG_LEVEL'])
    
    # Fonctions template globales
    @app.template_global()
    def format_datetime_global(dt):
        return format_datetime(dt)
    
    @app.template_global()
    def format_file_size_global(size_bytes):
        return format_file_size(size_bytes)
    
    @app.template_global()
    def truncate_text_global(text, length=50):
        return truncate_text(text, length)
    
    # Routes principales
    @app.route('/')
    def index():
        """Page d'accueil."""
        if current_user.is_authenticated:
            return redirect(url_for('main.dashboard'))
        
        # Statistiques pour la page d'accueil
        total_users = User.query.count()
        total_datasets = Dataset.query.count()
        total_analyses = AnalysisResult.query.count()
        
        stats = {
            'users': total_users,
            'datasets': total_datasets,
            'analyses': total_analyses,
            'data_processed': sum(d.get_file_size_mb() for d in Dataset.query.all())
        }
        
        return render_template('index.html', stats=stats)
    
    # Blueprint pour l'authentification
    from flask import Blueprint
    auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
    
    @auth_bp.route('/login', methods=['GET', 'POST'])
    @limiter.limit("10 per minute")
    def login():
        """Page de connexion."""
        if current_user.is_authenticated:
            return redirect(url_for('main.dashboard'))
        
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            
            if user and user.check_password(form.password.data):
                if not user.is_active:
                    flash('Votre compte est désactivé. Contactez l\'administrateur.', 'error')
                    return render_template('auth/login.html', form=form)
                
                login_user(user, remember=form.remember_me.data)
                user.last_login = datetime.utcnow()
                user.last_activity = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('main.dashboard')
                
                flash(f'Bienvenue, {user.get_full_name()}!', 'success')
                return redirect(next_page)
            else:
                flash('Email ou mot de passe incorrect.', 'error')
        
        return render_template('auth/login.html', form=form)
    
    @auth_bp.route('/register', methods=['GET', 'POST'])
    @limiter.limit("5 per minute")
    def register():
        """Page d'inscription."""
        if current_user.is_authenticated:
            return redirect(url_for('main.dashboard'))
        
        form = RegisterForm()
        if form.validate_on_submit():
            user = User(
                username=form.username.data,
                email=form.email.data,
                first_name=form.first_name.data,
                last_name=form.last_name.data
            )
            user.set_password(form.password.data)
            
            try:
                db.session.add(user)
                db.session.commit()
                
                flash('Inscription réussie! Vous pouvez maintenant vous connecter.', 'success')
                return redirect(url_for('auth.login'))
                
            except Exception as e:
                db.session.rollback()
                flash('Erreur lors de l\'inscription. Veuillez réessayer.', 'error')
                app.logger.error(f"Erreur d'inscription: {e}")
        
        return render_template('auth/register.html', form=form)
    
    @auth_bp.route('/logout')
    @login_required
    def logout():
        """Déconnexion."""
        current_user.last_activity = datetime.utcnow()
        db.session.commit()
        logout_user()
        flash('Vous êtes déconnecté.', 'info')
        return redirect(url_for('index'))
    
    app.register_blueprint(auth_bp)
    
    # Blueprint principal
    main_bp = Blueprint('main', __name__)
    
    @main_bp.route('/dashboard')
    @login_required
    def dashboard():
        """Dashboard principal."""
        # Statistiques utilisateur
        user_datasets = current_user.datasets.count()
        user_jobs = current_user.cleaning_jobs.count()
        quota_percentage = current_user.get_quota_percentage()
        
        # Datasets récents
        recent_datasets = current_user.datasets.order_by(Dataset.upload_date.desc()).limit(5).all()
        
        # Tâches récentes
        recent_jobs = current_user.cleaning_jobs.order_by(CleaningJob.created_at.desc()).limit(5).all()
        
        return render_template('dashboard.html',
                             datasets_count=user_datasets,
                             jobs_count=user_jobs,
                             quota_percentage=quota_percentage,
                             recent_datasets=recent_datasets,
                             recent_jobs=recent_jobs)
    
    @main_bp.route('/upload', methods=['GET', 'POST'])
    @login_required
    @limiter.limit("20 per hour")
    def upload_file():
        """Upload de fichier."""
        form = UploadForm()
        
        if form.validate_on_submit():
            file = form.file.data
            
            # Vérifier la taille du fichier
            file.seek(0, 2)  # Aller à la fin
            file_size = file.tell()
            file.seek(0)  # Retourner au début
            
            file_size_mb = file_size / (1024 * 1024)
            
            if not current_user.can_upload(file_size_mb):
                flash(f'Quota dépassé. Vous ne pouvez pas uploader un fichier de {file_size_mb:.1f} MB.', 'error')
                return render_template('upload.html', form=form)
            
            try:
                # Sauvegarder le fichier
                file_manager = FileManager(current_app.config['UPLOAD_FOLDER'])
                file_info = file_manager.save_file(file, current_user.id)
                
                if not file_info:
                    flash('Erreur lors de la sauvegarde du fichier.', 'error')
                    return render_template('upload.html', form=form)
                
                # Créer l'entrée dataset
                dataset = Dataset(
                    name=form.dataset_name.data,
                    filename=file_info['filename'],
                    file_path=file_info['file_path'],
                    file_type=file_info['filename'].rsplit('.', 1)[1].lower(),
                    file_size=file_info['file_size'],
                    user_id=current_user.id
                )
                
                # Charger et analyser le dataset
                processor = DataProcessor()
                df = processor.load_dataset(
                    file_info['file_path'],
                    dataset.file_type
                )
                
                # Mettre à jour les métadonnées
                dataset.update_metadata(df)
                
                # Sauvegarder en base
                db.session.add(dataset)
                current_user.update_quota(file_size_mb)
                db.session.commit()
                
                flash(f'Dataset "{dataset.name}" uploadé avec succès!', 'success')
                return redirect(url_for('main.view_dataset', dataset_id=dataset.id))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Erreur lors du traitement du fichier: {str(e)}', 'error')
                app.logger.error(f"Erreur upload dataset: {e}")
        
        return render_template('upload.html', form=form)
    
    @app.route('/nettoyage-auto')
    @login_required
    def nettoyage_auto():
        """Page de nettoyage automatique des données"""
        datasets = current_user.datasets.all()
        cleaning_jobs = current_user.cleaning_jobs.order_by(CleaningJob.created_at.desc()).limit(10).all()
        
        return render_template('Nettoyage_Auto.html', 
                             datasets=datasets, 
                             cleaning_jobs=cleaning_jobs)
    
    @app.route('/analyses')
    @login_required
    def analyses():
        """Page d'analyses avancées"""
        datasets = current_user.datasets.filter_by(is_cleaned=True).all()
        analyses = AnalysisResult.query.join(Dataset).filter(Dataset.user_id == current_user.id).all()
        
        return render_template('Analyses.html', 
                             datasets=datasets, 
                             analyses=analyses)
    
    @app.route('/etl-pipeline')
    @login_required
    def etl_pipeline():
        """Page de pipeline ETL"""
        pipelines = []  # À implémenter avec les pipelines ETL
        datasets = current_user.datasets.all()
        
        return render_template('ETL_Pipeline.html', 
                             pipelines=pipelines, 
                             datasets=datasets)
    
    @app.route('/ml-auto')
    @login_required
    def ml_auto():
        """Page de Machine Learning automatique"""
        models = []  # À implémenter avec les modèles ML
        datasets = current_user.datasets.filter_by(is_analyzed=True).all()
        
        return render_template('ML_Auto.html', 
                             models=models, 
                             datasets=datasets)
    
    @app.route('/parametres')
    @login_required
    def parametres():
        """Page de paramètres"""
        return render_template('Parametres.html', user=current_user)
    
    @app.route('/profil')
    @login_required
    def profil():
        """Page de profil utilisateur"""
        return render_template('Profil.html', user=current_user)
    
    @app.route('/api-keys')
    @login_required
    def api_keys():
        """Page de gestion des clés API"""
        tokens = current_user.api_tokens.all()
        return render_template('API_Keys.html', tokens=tokens, user=current_user)
    
    @app.route('/integrations')
    @login_required
    def integrations():
        """Page d'intégrations"""
        return render_template('Integrations.html')
    
    # API pour le nettoyage automatique
    @app.route('/api/start-cleaning', methods=['POST'])
    @login_required
    def start_cleaning():
        """Démarre un processus de nettoyage automatique"""
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        cleaning_mode = data.get('mode', 'automatic')
        
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()
        if not dataset:
            return jsonify({'error': 'Dataset non trouvé'}), 404
        
        # Créer une tâche de nettoyage
        job = CleaningJob(
            name=f"Nettoyage auto - {dataset.name}",
            status='running',
            cleaning_mode=cleaning_mode,
            user_id=current_user.id,
            dataset_id=dataset.id
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Simuler le nettoyage (à remplacer par la vraie logique)
        job.status = 'completed'
        job.progress = 100
        job.completed_at = datetime.utcnow()
        dataset.is_cleaned = True
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'job_id': job.id,
            'message': 'Nettoyage terminé avec succès'
        })
    
    # API pour l'analyse
    @app.route('/api/start-analysis', methods=['POST'])
    @login_required
    def start_analysis():
        """Démarre une analyse automatique"""
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        analysis_type = data.get('type', 'statistical')
        
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()
        if not dataset:
            return jsonify({'error': 'Dataset non trouvé'}), 404
        
        # Créer une analyse
        analysis = AnalysisResult(
            name=f"Analyse {analysis_type} - {dataset.name}",
            analysis_type=analysis_type,
            dataset_id=dataset.id,
            results_data=json.dumps({
                'type': analysis_type,
                'summary': 'Analyse terminée avec succès',
                'timestamp': datetime.utcnow().isoformat()
            })
        )
        
        db.session.add(analysis)
        dataset.is_analyzed = True
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'message': 'Analyse terminée avec succès'
        })
    
    # API pour créer des clés API
    @app.route('/api/create-token', methods=['POST'])
    @login_required
    def create_api_token():
        """Crée une nouvelle clé API"""
        data = request.get_json()
        token_name = data.get('name', 'Nouveau token')
        
        # Générer un token unique
        token_value = secrets.token_urlsafe(32)
        
        api_token = APIToken(
            name=token_name,
            token_hash=generate_password_hash(token_value),
            user_id=current_user.id
        )
        
        db.session.add(api_token)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'token': token_value,
            'token_id': api_token.id,
            'message': 'Token créé avec succès'
        })


    
    @main_bp.route('/datasets')
    @login_required
    def list_datasets():
        """Liste des datasets."""
        page = request.args.get('page', 1, type=int)
        search = request.args.get('search', '', type=str)
        file_type = request.args.get('type', '', type=str)
        
        query = current_user.datasets
        
        if search:
            query = query.filter(Dataset.name.contains(search))
        if file_type:
            query = query.filter(Dataset.file_type == file_type)
        
        datasets = query.order_by(Dataset.upload_date.desc()).paginate(
            page=page,
            per_page=current_app.config['DATASETS_PER_PAGE'],
            error_out=False
        )
        
        return render_template('datasets/list.html', datasets=datasets, search=search, file_type=file_type)
    
    @main_bp.route('/dataset/<int:dataset_id>')
    @login_required
    def view_dataset(dataset_id):
        """Vue détaillée d'un dataset."""
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        try:
            # Charger le dataset
            processor = DataProcessor()
            df = processor.load_dataset(dataset.file_path, dataset.file_type)
            
            # Analyse de qualité
            quality_analysis = processor.analyze_dataset_quality(df)
            
            # Échantillon de données
            sample_data = df.head(100).to_dict('records')
            
            # Informations sur les colonnes
            columns_info = dataset.get_columns_info()
            
            return render_template('datasets/view.html',
                                 dataset=dataset,
                                 sample_data=sample_data,
                                 columns_info=columns_info,
                                 basic_stats=quality_analysis)
            
        except Exception as e:
            flash(f'Erreur lors du chargement du dataset: {str(e)}', 'error')
            app.logger.error(f"Erreur chargement dataset {dataset_id}: {e}")
            return redirect(url_for('main.list_datasets'))
    
    @main_bp.route('/dataset/<int:dataset_id>/clean', methods=['GET', 'POST'])
    @login_required
    def clean_dataset(dataset_id):
        """Nettoyage d'un dataset."""
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        form = CleaningForm()
        form.job_name.data = form.job_name.data or f"Nettoyage de {dataset.name}"
        
        if form.validate_on_submit():
            # Créer la tâche de nettoyage
            job = CleaningJob(
                name=form.job_name.data,
                cleaning_mode=form.cleaning_mode.data,
                remove_duplicates=form.remove_duplicates.data,
                handle_missing_values=form.handle_missing_values.data,
                detect_outliers=form.detect_outliers.data,
                standardize_formats=form.standardize_formats.data,
                user_id=current_user.id,
                dataset_id=dataset.id
            )
            
            db.session.add(job)
            db.session.commit()
            
            # Lancer le nettoyage en arrière-plan
            try:
                cleaning_config = {
                    'remove_duplicates': form.remove_duplicates.data,
                    'handle_missing_values': form.handle_missing_values.data,
                    'detect_outliers': form.detect_outliers.data,
                    'outlier_method': form.outlier_method.data,
                    'standardize_formats': form.standardize_formats.data,
                    'normalize_text': form.normalize_text.data,
                    'convert_data_types': form.convert_data_types.data
                }
                
                processor = DataProcessor()
                cleaner = DataCleaner()
                
                # Charger le dataset
                job.status = 'running'
                job.started_at = datetime.utcnow()
                db.session.commit()
                
                df = processor.load_dataset(dataset.file_path, dataset.file_type)
                job.original_rows = len(df)
                
                # Nettoyer
                cleaned_df, cleaning_report = cleaner.clean_dataset(df, cleaning_config, job)
                
                # Sauvegarder le dataset nettoyé
                if form.backup_original.data:
                    # Créer une sauvegarde
                    import shutil
                    backup_path = dataset.file_path + '.backup'
                    shutil.copy2(dataset.file_path, backup_path)
                
                # Sauvegarder le dataset nettoyé (selon le format original)
                if dataset.file_type == 'csv':
                    cleaned_df.to_csv(dataset.file_path, index=False)
                elif dataset.file_type in ['xlsx', 'xls']:
                    cleaned_df.to_excel(dataset.file_path, index=False)
                elif dataset.file_type == 'json':
                    cleaned_df.to_json(dataset.file_path, orient='records', indent=2)
                
                # Mettre à jour les métadonnées
                dataset.update_metadata(cleaned_df)
                dataset.is_cleaned = True
                dataset.last_cleaned = datetime.utcnow()
                
                # Finaliser la tâche
                job.status = 'completed'
                job.completed_at = datetime.utcnow()
                job.cleaned_rows = len(cleaned_df)
                job.removed_duplicates = cleaning_report.get('rows_removed', 0)
                
                db.session.commit()
                
                flash(f'Dataset nettoyé avec succès! {job.removed_duplicates} lignes supprimées.', 'success')
                return redirect(url_for('main.view_cleaning_job', job_id=job.id))
                
            except Exception as e:
                job.status = 'failed'
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.session.commit()
                
                flash(f'Erreur lors du nettoyage: {str(e)}', 'error')
                app.logger.error(f"Erreur nettoyage dataset {dataset_id}: {e}")
        
        return render_template('datasets/clean.html', dataset=dataset, form=form)
    
    @main_bp.route('/dataset/<int:dataset_id>/analyze', methods=['GET', 'POST'])
    @login_required
    def analyze_dataset(dataset_id):
        """Analyse d'un dataset."""
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        
        form = AnalysisForm()
        
        # Peupler les choix de colonnes
        try:
            processor = DataProcessor()
            df = processor.load_dataset(dataset.file_path, dataset.file_type)
            
            column_choices = [(col, col) for col in df.columns]
            form.target_column.choices = [('', 'Aucune')] + column_choices
            form.feature_columns.choices = column_choices
            
        except Exception as e:
            flash(f'Erreur lors du chargement du dataset: {str(e)}', 'error')
            return redirect(url_for('main.view_dataset', dataset_id=dataset_id))
        
        if form.validate_on_submit():
            try:
                # Lancer l'analyse
                analyzer = MLAnalyzer()
                viz_generator = VisualizationGenerator()
                insight_generator = InsightGenerator()
                
                target_col = form.target_column.data if form.target_column.data else None
                analysis_results = analyzer.auto_analyze(df, target_col, form.analysis_types.data)
                
                # Générer les visualisations
                if form.include_visualizations.data:
                    visualizations = viz_generator.generate_visualizations(df, analysis_results, form.analysis_types.data)
                    analysis_results['visualizations'] = visualizations
                
                # Générer les insights
                if form.generate_insights.data:
                    insights = insight_generator.generate_insights(analysis_results, df)
                    analysis_results['insights'] = insights
                
                # Sauvegarder l'analyse
                analysis = AnalysisResult(
                    name=form.analysis_name.data,
                    analysis_type=form.analysis_types.data,
                    dataset_id=dataset.id
                )
                analysis.set_results_data(analysis_results)
                
                if 'visualizations' in analysis_results:
                    analysis.set_visualizations(analysis_results['visualizations'])
                
                if 'insights' in analysis_results:
                    analysis.set_insights(analysis_results['insights'])
                
                db.session.add(analysis)
                dataset.is_analyzed = True
                dataset.last_analyzed = datetime.utcnow()
                db.session.commit()
                
                flash('Analyse terminée avec succès!', 'success')
                return redirect(url_for('main.view_analysis', analysis_id=analysis.id))
                
            except Exception as e:
                flash(f'Erreur lors de l\'analyse: {str(e)}', 'error')
                app.logger.error(f"Erreur analyse dataset {dataset_id}: {e}")
        
        return render_template('datasets/analyze.html', dataset=dataset, form=form)
    
    @main_bp.route('/job/<int:job_id>')
    @login_required
    def view_cleaning_job(job_id):
        """Vue détaillée d'une tâche de nettoyage."""
        job = CleaningJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()
        return render_template('jobs/view.html', job=job)
    
    @main_bp.route('/analysis/<int:analysis_id>')
    @login_required
    def view_analysis(analysis_id):
        """Vue détaillée d'une analyse."""
        analysis = AnalysisResult.query.filter_by(id=analysis_id).first_or_404()
        
        # Vérifier que l'utilisateur a accès à cette analyse
        if analysis.dataset.user_id != current_user.id:
            abort(403)
        
        return render_template('analysis/view.html', analysis=analysis)
    
    @main_bp.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        """Profil utilisateur."""
        form = ProfileForm(original_email=current_user.email)
        
        if form.validate_on_submit():
            current_user.first_name = form.first_name.data
            current_user.last_name = form.last_name.data
            current_user.email = form.email.data
            current_user.default_cleaning_mode = form.default_cleaning_mode.data
            current_user.auto_analysis_enabled = form.auto_analysis_enabled.data
            
            db.session.commit()
            flash('Profil mis à jour avec succès!', 'success')
            return redirect(url_for('main.profile'))
        elif request.method == 'GET':
            form.first_name.data = current_user.first_name
            form.last_name.data = current_user.last_name
            form.email.data = current_user.email
            form.default_cleaning_mode.data = current_user.default_cleaning_mode
            form.auto_analysis_enabled.data = current_user.auto_analysis_enabled
        
        return render_template('auth/profile.html', form=form)
    
    @main_bp.route('/change-password', methods=['GET', 'POST'])
    @login_required
    def change_password():
        """Changement de mot de passe."""
        form = PasswordChangeForm()
        
        if form.validate_on_submit():
            if current_user.check_password(form.current_password.data):
                current_user.set_password(form.new_password.data)
                db.session.commit()
                flash('Mot de passe modifié avec succès!', 'success')
                return redirect(url_for('main.profile'))
            else:
                flash('Mot de passe actuel incorrect.', 'error')
        
        return render_template('auth/change_password.html', form=form)
    
    @main_bp.route('/export/<int:dataset_id>')
    @login_required
    def export_dataset(dataset_id):
        """Export d'un dataset."""
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
        format_type = request.args.get('format', 'csv')
        
        try:
            processor = DataProcessor()
            df = processor.load_dataset(dataset.file_path, dataset.file_type)
            
            # Créer le fichier d'export
            output = BytesIO()
            
            if format_type == 'csv':
                df.to_csv(output, index=False)
                mimetype = 'text/csv'
                filename = f"{dataset.name}.csv"
            elif format_type == 'excel':
                df.to_excel(output, index=False)
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                filename = f"{dataset.name}.xlsx"
            elif format_type == 'json':
                df.to_json(output, orient='records', indent=2)
                mimetype = 'application/json'
                filename = f"{dataset.name}.json"
            else:
                flash('Format d\'export non supporté.', 'error')
                return redirect(url_for('main.view_dataset', dataset_id=dataset_id))
            
            output.seek(0)
            
            return send_file(
                output,
                mimetype=mimetype,
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            flash(f'Erreur lors de l\'export: {str(e)}', 'error')
            app.logger.error(f"Erreur export dataset {dataset_id}: {e}")
            return redirect(url_for('main.view_dataset', dataset_id=dataset_id))
    
    app.register_blueprint(main_bp)
    
    # API Blueprint
    api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
    
    def require_api_key(f):
        """Décorateur pour vérifier la clé API."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if not api_key:
                return jsonify({'error': 'Clé API requise'}), 401
            
            token = APIToken.query.filter_by(token_hash=generate_password_hash(api_key)).first()
            if not token or not token.is_active or token.is_expired():
                return jsonify({'error': 'Clé API invalide ou expirée'}), 401
            
            token.record_usage()
            request.current_token = token
            
            return f(*args, **kwargs)
        return decorated_function
    
    @api_bp.route('/datasets', methods=['GET'])
    @require_api_key
    def api_list_datasets():
        """API: Liste des datasets."""
        user_id = request.current_token.user_id
        datasets = Dataset.query.filter_by(user_id=user_id).all()
        
        return jsonify({
            'datasets': [dataset.to_dict() for dataset in datasets],
            'count': len(datasets)
        })
    
    @api_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
    @require_api_key
    def api_get_dataset(dataset_id):
        """API: Détails d'un dataset."""
        user_id = request.current_token.user_id
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first_or_404()
        
        return jsonify(dataset.to_dict())
    
    @api_bp.route('/datasets/<int:dataset_id>/data', methods=['GET'])
    @require_api_key
    def api_get_dataset_data(dataset_id):
        """API: Données d'un dataset."""
        user_id = request.current_token.user_id
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first_or_404()
        
        try:
            processor = DataProcessor()
            df = processor.load_dataset(dataset.file_path, dataset.file_type)
            
            # Pagination
            page = request.args.get('page', 1, type=int)
            per_page = min(request.args.get('per_page', 100, type=int), 1000)
            
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            
            data = df.iloc[start_idx:end_idx].to_dict('records')
            
            return jsonify({
                'data': data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': len(df),
                    'pages': (len(df) + per_page - 1) // per_page
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/datasets/<int:dataset_id>/analyze', methods=['POST'])
    @require_api_key
    def api_analyze_dataset(dataset_id):
        """API: Analyse d'un dataset."""
        user_id = request.current_token.user_id
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first_or_404()
        
        try:
            data = request.get_json()
            analysis_type = data.get('analysis_type', 'statistical')
            target_column = data.get('target_column')
            
            processor = DataProcessor()
            df = processor.load_dataset(dataset.file_path, dataset.file_type)
            
            analyzer = MLAnalyzer()
            results = analyzer.auto_analyze(df, target_column, analysis_type)
            
            return jsonify({
                'analysis_results': results,
                'dataset_id': dataset_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @api_bp.route('/health', methods=['GET'])
    def api_health():
        """API: Statut de santé."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0'
        })
    
    app.register_blueprint(api_bp)
    
    # Gestionnaires d'erreurs
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return render_template('errors/429.html'), 429
    
    return app

# Point d'entrée principal
if __name__ == '__main__':
    app = create_app(os.environ.get('FLASK_ENV', 'development'))
    
    with app.app_context():
        # Créer les tables si elles n'existent pas
        db.create_all()
        
        # Créer un utilisateur admin par défaut
        admin = User.query.filter_by(email='admin@datacleaner.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@datacleaner.com',
                first_name='Admin',
                last_name='DataCleaner',
                is_admin=True,
                is_premium=True,
                quota_limit=10000.0  # 10GB pour l'admin
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Utilisateur admin créé: admin@datacleaner.com / admin123")
    
    app.run(debug=True, host='0.0.0.0', port=5000)