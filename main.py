
from flask import Flask, jsonify, request, abort
from werkzeug.security import generate_password_hash, check_password_hash
from app import create_app, db
from app.models import User, Project, Citation

app = create_app()

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return jsonify({"message": "Literature Review API is running"})

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    if not all(k in data for k in ["first_name", "last_name", "email", "password"]):
        return jsonify({"error": "Missing required fields"}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already registered"}), 400
        
    user = User(
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        password=generate_password_hash(data['password'])
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({"message": "User created successfully"}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not all(k in data for k in ["email", "password"]):
        return jsonify({"error": "Missing email or password"}), 400
        
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({"error": "Invalid email or password"}), 401
        
    return jsonify({
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email
        }
    })

@app.route('/api/projects', methods=['GET'])
def get_projects():
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    projects = Project.query.filter_by(user_id=user_id).all()
    return jsonify({
        "projects": [{
            "id": p.id,
            "name": p.name,
            "created_at": p.created_at,
            "current_iteration": p.current_iteration
        } for p in projects]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
