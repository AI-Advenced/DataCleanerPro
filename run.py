#!/usr/bin/env python3
"""
DataCleaner-Pro Application Runner
==================================

Script principal pour lancer l'application Flask DataCleaner-Pro.

Usage:
    python run.py                    # Mode développement
    python run.py --production       # Mode production
    python run.py --host 0.0.0.0 --port 8000  # Personnalisé
"""

import os
import sys
import click
import logging
from datetime import datetime

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db, User
from config import get_config

def setup_logging(app):
    """Configure logging for the application."""
    if not app.debug and not app.testing:
        # Production logging
        if app.config.get('LOG_FILE'):
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                app.config['LOG_FILE'],
                maxBytes=app.config.get('LOG_MAX_BYTES', 10 * 1024 * 1024),
                backupCount=app.config.get('LOG_BACKUP_COUNT', 5)
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, app.config.get('LOG_LEVEL', 'INFO')))
            app.logger.addHandler(file_handler)
        
        app.logger.setLevel(getattr(logging, app.config.get('LOG_LEVEL', 'INFO')))
        app.logger.info('DataCleaner-Pro startup')

def create_admin_user(app):
    """Create default admin user if it doesn't exist."""
    with app.app_context():
        admin = User.query.filter_by(email='admin@datacleaner.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@datacleaner.com',
                first_name='Admin',
                last_name='DataCleaner',
                is_admin=True,
                is_premium=True,
                quota_limit=app.config.get('DEFAULT_ENTERPRISE_QUOTA_MB', 1000 * 1024)
            )
            admin.set_password('admin123')
            
            try:
                db.session.add(admin)
                db.session.commit()
                print(f"✅ Admin user created: admin@datacleaner.com / admin123")
            except Exception as e:
                db.session.rollback()
                print(f"❌ Error creating admin user: {e}")

def init_database(app):
    """Initialize the database and create tables."""
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully")
            
            # Create admin user
            create_admin_user(app)
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            sys.exit(1)

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'plotly', 'sqlalchemy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All required packages are installed")

def display_banner():
    """Display application banner."""
    banner = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      DataCleaner-Pro                          ║
    ║                   Version 2.0.0 - Advanced                   ║
    ║                                                              ║
    ║        Plateforme IA de Nettoyage et d'Analyse de Données   ║
    ║                                                              ║
    ║    • Nettoyage automatique intelligent                      ║
    ║    • Analyses statistiques complètes                        ║
    ║    • Machine Learning automatique                           ║
    ║    • Visualisations interactives                            ║
    ║    • API REST complète                                      ║
    ║                                                              ║
    ║    Démarré le: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

@click.command()
@click.option('--host', '-h', default='127.0.0.1', help='Host address')
@click.option('--port', '-p', default=5000, help='Port number')
@click.option('--debug/--no-debug', default=None, help='Enable debug mode')
@click.option('--production', is_flag=True, help='Run in production mode')
@click.option('--init-db', is_flag=True, help='Initialize database only')
@click.option('--create-admin', is_flag=True, help='Create admin user only')
@click.option('--check-deps', is_flag=True, help='Check dependencies only')
def main(host, port, debug, production, init_db, create_admin, check_deps):
    """Launch DataCleaner-Pro application."""
    
    # Display banner
    display_banner()
    
    # Check dependencies if requested
    if check_deps:
        check_requirements()
        return
    
    # Determine configuration
    if production:
        config_name = 'production'
        if debug is None:
            debug = False
    else:
        config_name = os.environ.get('FLASK_ENV', 'development')
        if debug is None:
            debug = config_name == 'development'
    
    print(f"🔧 Configuration: {config_name}")
    print(f"🐛 Debug mode: {'enabled' if debug else 'disabled'}")
    
    # Check dependencies
    check_requirements()
    
    # Create application
    try:
        app = create_app(config_name)
        print("✅ Application created successfully")
    except Exception as e:
        print(f"❌ Error creating application: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(app)
    
    # Initialize database if requested or if it doesn't exist
    db_path = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    if init_db or ('sqlite' in db_path and not os.path.exists(db_path.replace('sqlite:///', ''))):
        print("🗃️  Initializing database...")
        init_database(app)
    
    # Create admin user if requested
    if create_admin:
        print("👤 Creating admin user...")
        create_admin_user(app)
        return
    
    # If only database operations were requested, exit
    if init_db:
        print("✅ Database initialization complete")
        return
    
    # Display application information
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📊 Admin Panel: http://{host}:{port}/admin (admin@datacleaner.com / admin123)")
    print(f"📚 API Documentation: http://{host}:{port}/api/v1/docs")
    print("🚀 Application is ready!")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 65)
    
    # Start the application
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=debug
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()