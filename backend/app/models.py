
from . import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    projects = db.relationship('Project', backref='user', lazy=True)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    citations = db.relationship('Citation', backref='project', lazy=True)
    current_iteration = db.Column(db.Integer, default=0)
    keywords = db.Column(db.JSON, default=lambda: {"include": [], "exclude": []})
    model_metrics = db.Column(db.JSON, default=dict)
    
    # New fields for duplicate removal flow
    duplicates_removed = db.Column(db.Boolean, default=False)
    duplicates_count = db.Column(db.Integer, default=0)
    keywords_selected = db.Column(db.Boolean, default=False)
    citations_count = db.Column(db.Integer, default=0)

class Citation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    abstract = db.Column(db.Text, nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    is_relevant = db.Column(db.Boolean, nullable=True)
    iteration = db.Column(db.Integer, default=0)
    is_duplicate = db.Column(db.Boolean, default=False)
