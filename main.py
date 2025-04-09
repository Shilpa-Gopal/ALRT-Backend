
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

@app.route('/api/projects', methods=['POST'])
def create_project():
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    if 'name' not in data:
        return jsonify({"error": "Project name is required"}), 400
        
    project = Project(
        name=data['name'],
        user_id=user_id,
        keywords={"include": [], "exclude": []}
    )
    db.session.add(project)
    db.session.commit()
    
    return jsonify({
        "project": {
            "id": project.id,
            "name": project.name,
            "created_at": project.created_at,
            "current_iteration": project.current_iteration
        }
    }), 201

@app.route('/api/projects/<int:project_id>/citations', methods=['POST'])
def add_citations(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    data = request.get_json()
    if not isinstance(data.get('citations'), list):
        return jsonify({"error": "Citations must be a list"}), 400
        
    new_citations = []
    for citation_data in data['citations']:
        citation = Citation(
            title=citation_data['title'],
            abstract=citation_data['abstract'],
            project_id=project_id,
            iteration=project.current_iteration
        )
        new_citations.append(citation)
        
    db.session.bulk_save_objects(new_citations)
    db.session.commit()
    
    return jsonify({"message": f"Added {len(new_citations)} citations"}), 201

@app.route('/api/projects/<int:project_id>/citations/<int:citation_id>', methods=['PUT'])
def update_citation(project_id, citation_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    citation = Citation.query.filter_by(id=citation_id, project_id=project_id).first()
    if not citation:
        return jsonify({"error": "Citation not found"}), 404
        
    data = request.get_json()
    if 'is_relevant' in data:
        citation.is_relevant = data['is_relevant']
        db.session.commit()
        
    return jsonify({
        "citation": {
            "id": citation.id,
            "title": citation.title,
            "is_relevant": citation.is_relevant
        }
    })

@app.route('/api/projects/<int:project_id>/train', methods=['POST'])
def train_model(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    review_system = LiteratureReviewSystem(project_id)
    result = review_system.train_iteration(project.current_iteration)
    
    if 'error' in result:
        return jsonify(result), 400
        
    project.model_metrics[str(project.current_iteration)] = result['metrics']
    project.current_iteration += 1
    db.session.commit()
    
    return jsonify(result)

@app.route('/api/projects/<int:project_id>/keywords', methods=['GET'])
def get_keywords(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    return jsonify(project.keywords)

@app.route('/api/projects/<int:project_id>/keywords', methods=['PUT'])
def update_keywords(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    data = request.get_json()
    if not isinstance(data, dict) or not all(k in data for k in ['include', 'exclude']):
        return jsonify({"error": "Invalid keywords format"}), 400
        
    project.keywords = data
    db.session.commit()
    
    return jsonify(project.keywords)

@app.route('/api/projects/<int:project_id>/citations/filter', methods=['GET'])
def filter_citations(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
        
    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
        
    iteration = request.args.get('iteration', type=int)
    is_relevant = request.args.get('is_relevant', type=lambda v: v.lower() == 'true' if v else None)
    
    query = Citation.query.filter_by(project_id=project_id)
    if iteration is not None:
        query = query.filter_by(iteration=iteration)
    if is_relevant is not None:
        query = query.filter_by(is_relevant=is_relevant)
        
    citations = query.all()
    return jsonify({
        "citations": [{
            "id": c.id,
            "title": c.title,
            "abstract": c.abstract,
            "is_relevant": c.is_relevant,
            "iteration": c.iteration
        } for c in citations]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
