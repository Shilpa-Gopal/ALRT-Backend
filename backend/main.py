from flask import Flask, jsonify, request, abort, send_file
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
import xlsxwriter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.security import generate_password_hash, check_password_hash
from app import create_app, db
from app.models import User, Project, Citation

app = create_app()
CORS(app, resources={r"/api/*": {"origins": "*"}})

with app.app_context():
    db.create_all()


@app.route('/', methods=['GET'])
def home():
    return 'Backend API is running', 200


@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()

    if not all(k in data
               for k in ["first_name", "last_name", "email", "password"]):
        return jsonify({"error": "Missing required fields"}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already registered"}), 400

    user = User(first_name=data['first_name'],
                last_name=data['last_name'],
                email=data['email'],
                password=generate_password_hash(data['password']))

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
    try:
        app.logger.info("Received project creation request")
        app.logger.info(f"Headers: {dict(request.headers)}")
        
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            app.logger.error("No user ID in request headers")
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        app.logger.info(f"Request data: {data}")
        
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
            
        if 'name' not in data:
            app.logger.error("No project name in request data")
            return jsonify({"error": "Project name is required"}), 400

        project = Project(name=data['name'],
                        user_id=user_id,
                        keywords={
                            "include": [],
                            "exclude": []
                        })
        db.session.add(project)
        db.session.commit()
        
        app.logger.info(f"Project created successfully with ID: {project.id}")
        return jsonify({
            "project": {
                "id": project.id,
                "name": project.name,
                "created_at": project.created_at,
                "current_iteration": project.current_iteration
            }
        }), 201
        
    except Exception as e:
        app.logger.error(f"Project creation error: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Failed to create project", "details": str(e)}), 500


@app.route('/api/projects/<int:project_id>/citations', methods=['POST'])
def add_citations(project_id):
    try:
        app.logger.info(f"Received citation upload request for project {project_id}")
        app.logger.info(f"Request headers: {dict(request.headers)}")
        app.logger.info(f"Request files: {request.files}")
        app.logger.info(f"Request form: {request.form}")
        
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            app.logger.error("No user ID in request headers")
            return jsonify({"error": "Unauthorized"}), 401

        project = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if not project:
            app.logger.error(f"Project {project_id} not found for user {user_id}")
            return jsonify({"error": "Project not found"}), 404

        if 'file' not in request.files:
            app.logger.info("No file in request, checking for JSON data")
            data = request.get_json()
            if not isinstance(data.get('citations'), list):
                return jsonify({"error": "Citations must be a list"}), 400

            new_citations = []
            for citation_data in data['citations']:
                citation = Citation(title=citation_data['title'],
                                    abstract=citation_data['abstract'],
                                    project_id=project_id,
                                    iteration=project.current_iteration)
                new_citations.append(citation)
        else:
            file = request.files['file']
            app.logger.info(f"Received file: {file.filename if file else 'No file'}")
            
            if not file or file.filename == '':
                app.logger.error("No file provided or empty filename")
                return jsonify({"error": "No file provided"}), 400

            if not file.filename.endswith(('.csv', '.xlsx')):
                app.logger.error(f"Invalid file format: {file.filename}")
                return jsonify({
                    "error": "Invalid file format. Only CSV and Excel files are supported.",
                    "filename": file.filename
                }), 400

            app.logger.info(f"Processing file: {file.filename} ({file.content_length} bytes)")

            try:
                app.logger.info(f"Attempting to process file: {file.filename}")
                # Save the file temporarily
                temp_path = os.path.join('/tmp', secure_filename(file.filename))
                file.save(temp_path)
                app.logger.info(f"Saved file temporarily: {temp_path}")
                
                try:
                    if file.filename.endswith('.csv'):
                        app.logger.info("Processing CSV file")
                        df = pd.read_csv(temp_path)
                    elif file.filename.endswith('.xlsx'):
                        app.logger.info("Processing Excel file")
                        df = pd.read_excel(temp_path, engine='openpyxl')
                        app.logger.info(f"Excel file read successfully with {len(df)} rows")
                    else:
                        app.logger.error(f"Unsupported file format: {file.filename}")
                        return jsonify({
                            "error": "Unsupported file format",
                            "details": f"File {file.filename} is not supported. Only .csv and .xlsx files are allowed"
                        }), 400
                    
                    # Clean up temp file
                    os.remove(temp_path)
                except Exception as e:
                    app.logger.error(f"File read error: {str(e)}")
                    return jsonify({
                        "error": "Failed to read file",
                        "details": str(e)
                    }), 400

                if df.empty:
                    return jsonify({"error": "File contains no data"}), 400

                if 'title' not in df.columns or 'abstract' not in df.columns:
                    return jsonify({
                        "error": "File must contain 'title' and 'abstract' columns",
                        "found_columns": list(df.columns)
                    }), 400

                new_citations = []
                for _, row in df.iterrows():
                    citation = Citation(title=str(row['title']),
                                     abstract=str(row['abstract']),
                                     project_id=project_id,
                                     iteration=project.current_iteration)
                    new_citations.append(citation)

                if not new_citations:
                    return jsonify({"error": "No valid citations found in file"}), 400

                db.session.bulk_save_objects(new_citations)
                db.session.commit()
                
                return jsonify({"message": f"Added {len(new_citations)} citations"}), 201

            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Upload processing error: {str(e)}")
                return jsonify({
                    "error": "Failed to process upload",
                    "details": str(e)
                }), 500

    except Exception as e:
        app.logger.error(f"General error in add_citations: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route('/api/projects/<int:project_id>/citations/<int:citation_id>',
           methods=['PUT'])
def update_citation(project_id, citation_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    citation = Citation.query.filter_by(id=citation_id,
                                        project_id=project_id).first()
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

    citations = Citation.query.filter_by(project_id=project_id).all()
    if not citations:
        return jsonify({"error": "No citations found"}), 404

    # Combine titles and abstracts for TF-IDF
    texts = [f"{c.title} {c.abstract}" for c in citations]
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get feature names and their scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    # Sort keywords by score
    keywords = [{
        "word": word,
        "score": float(score)
    } for word, score in zip(feature_names, scores)]
    keywords.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({
        "suggested_keywords": keywords[:50],
        "selected_keywords": project.keywords
    })


@app.route('/api/projects/<int:project_id>/keywords', methods=['PUT'])
def update_keywords(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    data = request.get_json()
    if not isinstance(data, dict) or not all(k in data
                                             for k in ['include', 'exclude']):
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
    is_relevant = request.args.get('is_relevant',
                                   type=lambda v: v.lower() == 'true'
                                   if v else None)

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


@app.route('/api/projects/<int:project_id>/download', methods=['GET'])
def download_results(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    citations = Citation.query.filter_by(project_id=project_id).all()

    output = io.BytesIO()
    from openpyxl import Workbook
    workbook = Workbook()
    worksheet = workbook.active

    # Write headers
    headers = [
        'Title', 'Abstract', 'Is Relevant', 'Iteration', 'Relevance Score'
    ]
    for col, header in enumerate(headers, start=1):
        worksheet.cell(row=1, column=col, value=header)

    # Get relevance scores for citations
    review_system = LiteratureReviewSystem(project_id)
    predictions = review_system.predict_relevance([{
        'title': c.title,
        'abstract': c.abstract
    } for c in citations])

    # Write data
    for row, (citation, prediction) in enumerate(zip(citations, predictions),
                                                 start=2):
        worksheet.cell(row=row, column=1, value=citation.title)
        worksheet.cell(row=row, column=2, value=citation.abstract)
        worksheet.cell(
            row=row,
            column=3,
            value='Yes' if citation.is_relevant else
            'No' if citation.is_relevant is not None else 'Unclassified')
        worksheet.cell(row=row, column=4, value=citation.iteration)
        worksheet.cell(row=row,
                       column=5,
                       value=prediction.get('relevance_probability', 0)
                       if 'error' not in prediction else 0)

    workbook.save(output)
    output.seek(0)

    return send_file(
        output,
        mimetype=
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'project_{project_id}_results.xlsx')


@app.route('/api/projects/<int:project_id>/predict', methods=['POST'])
def predict_citations(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    data = request.get_json()
    if not isinstance(data.get('citations'), list):
        return jsonify({"error": "Citations must be a list"}), 400

    review_system = LiteratureReviewSystem(project_id)
    predictions = review_system.predict_relevance(data['citations'])

    return jsonify({"predictions": predictions})


# At the end of your main.py file:
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


@app.route('/api/projects/<int:project_id>/iterations', methods=['GET'])
def get_iteration_info(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    return jsonify({
        "current_iteration": project.current_iteration,
        "max_iterations": 10,
        "metrics": project.model_metrics
    })


@app.route('/api/projects/<int:project_id>/labeled-citations', methods=['GET'])
def get_labeled_citations(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    labeled = Citation.query.filter(Citation.project_id == project_id,
                                    Citation.is_relevant.isnot(None)).all()

    return jsonify({
        "labeled_citations": [{
            "id": c.id,
            "title": c.title,
            "is_relevant": c.is_relevant,
            "iteration": c.iteration
        } for c in labeled]
    })