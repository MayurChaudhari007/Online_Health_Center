# /project/models.py

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
# from .extensions import db

# Create the database instance, but don't attach the app yet
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(350), unique=True, nullable=False)
    password_hash = db.Column(db.String(350), nullable=False)
    
    chats = db.relationship("Chat", backref="user", cascade="all, delete-orphan")
    reports = db.relationship("MedicalReport", backref="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    sender = db.Column(db.String(400))
    message = db.Column(db.Text)

class MedicalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_name = db.Column(db.String(260), nullable=False)
    mimetype = db.Column(db.String(100))
    size = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author_name = db.Column(db.String(100), nullable=False, default="Admin")
    created_at = db.Column(db.DateTime, server_default=func.now(), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    youtube_url = db.Column(db.String(500), nullable=True)