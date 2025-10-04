# /project/__init__.py

from flask import Flask
from .config import Config
from .models import db
from .routes import main_bp
from .extensions import mail

def create_app(config_class=Config):
    # Create and configure the app
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    mail.init_app(app)

    # Register blueprints
    app.register_blueprint(main_bp)

    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    # --- Jinja Filters (if any) ---
    @app.template_filter("filesize")
    def filesize(num):
        try:
            num = float(num)
        except (ValueError, TypeError):
            return "—"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024.0:
                return f"{num:3.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PB"

    return app