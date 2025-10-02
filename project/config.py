# /project/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "a_default_fallback_key_for_dev")
    
    # SQLAlchemy Config
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL2")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {"sslmode": "require"}
    }
    
    # File Upload Config
    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "doc", "docx"}
    
    # Gemini API Key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")