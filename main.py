
from flask import Flask, jsonify
from app import create_app, db
from app.models import User, Project, Citation

app = create_app()

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return jsonify({"message": "Literature Review API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
