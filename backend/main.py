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
from app.ml_system import LiteratureReviewSystem
from difflib import SequenceMatcher

app = create_app()

def calculate_similarity(text1, text2):
    """Calculate similarity ratio between two text strings"""
    if pd.isna(text1) or pd.isna(text2):
        return 0.0
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def find_year_column(df):
    """Find year-related column in dataframe (case insensitive)"""
    year_patterns = ['year', 'publication_year', 'pub_year', 'published_year', 
                    'publication_date', 'pub_date', 'date', 'published']
    
    for col in df.columns:
        for pattern in year_patterns:
            if pattern.lower() in col.lower():
                return col
    return None

def extract_year_from_value(value):
    """Extract year from various date formats"""
    if pd.isna(value):
        return None
    
    value_str = str(value)
    # Try to extract 4-digit year
    import re
    year_match = re.search(r'\b(19|20)\d{2}\b', value_str)
    if year_match:
        return int(year_match.group())
    return None

def calculate_completeness_score(row):
    """Calculate completeness score based on filled fields and content length"""
    score = 0
    total_fields = len(row)
    filled_fields = sum(1 for val in row if pd.notna(val) and str(val).strip())
    
    # Base score from field completion
    score += (filled_fields / total_fields) * 50
    
    # Additional score for abstract length
    if pd.notna(row.get('abstract')):
        abstract_len = len(str(row['abstract']))
        score += min(abstract_len / 10, 50)  # Up to 50 points for abstract length
    
    return score

def process_duplicates_and_create_citations(df, project_id, current_iteration):
    """Process duplicates using multiple strategies and create Citation objects"""
    
    # Normalize column names to lowercase for case-insensitive matching
    df_normalized = df.copy()
    df_normalized.columns = df_normalized.columns.str.lower()
    
    # Check required columns (case insensitive)
    if 'title' not in df_normalized.columns or 'abstract' not in df_normalized.columns:
        return {
            'error': f"File must contain 'title' and 'abstract' columns. Found: {list(df.columns)}",
            'citations': [],
            'duplicates_removed': 0,
            'removal_strategy': '',
            'duplicate_details': []
        }
    
    # Find year column
    year_col = find_year_column(df)
    has_year_data = year_col is not None
    
    # Find authors column (optional)
    authors_col = None
    for col in df.columns:
        if 'author' in col.lower():
            authors_col = col
            break
    
    duplicates = []
    duplicate_groups = []
    citations_to_keep = []
    removal_strategy = ""
    
    # Group potential duplicates
    for i in range(len(df)):
        current_row = df.iloc[i]
        is_duplicate = False
        
        for j, existing_row in enumerate(citations_to_keep):
            # Calculate similarity for title and abstract
            title_sim = calculate_similarity(current_row['title'], existing_row['title'])
            abstract_sim = calculate_similarity(current_row['abstract'], existing_row['abstract'])
            
            # Calculate author similarity if available
            author_sim = 1.0  # Default if no authors
            if authors_col and pd.notna(current_row.get(authors_col)) and pd.notna(existing_row.get(authors_col)):
                author_sim = calculate_similarity(current_row[authors_col], existing_row[authors_col])
            
            # Check if it's a duplicate (70% threshold)
            if title_sim >= 0.7 and abstract_sim >= 0.7 and author_sim >= 0.7:
                is_duplicate = True
                
                # Determine which one to keep based on strategy
                if has_year_data:
                    # Strategy 1: Keep latest by year
                    current_year = extract_year_from_value(current_row.get(year_col))
                    existing_year = extract_year_from_value(existing_row.get(year_col))
                    
                    if current_year and existing_year:
                        if current_year > existing_year:
                            # Replace existing with current
                            duplicate_groups.append({
                                'kept': f"{current_row['title'][:50]}...",
                                'removed': f"{existing_row['title'][:50]}...",
                                'reason': f'Newer publication year ({current_year} vs {existing_year})'
                            })
                            citations_to_keep[j] = current_row
                        else:
                            # Keep existing, mark current as duplicate
                            duplicate_groups.append({
                                'kept': f"{existing_row['title'][:50]}...",
                                'removed': f"{current_row['title'][:50]}...",
                                'reason': f'Older publication year ({current_year} vs {existing_year})'
                            })
                    elif current_year and not existing_year:
                        # Current has year, existing doesn't - keep current
                        duplicate_groups.append({
                            'kept': f"{current_row['title'][:50]}...",
                            'removed': f"{existing_row['title'][:50]}...",
                            'reason': 'Has publication year data'
                        })
                        citations_to_keep[j] = current_row
                    elif not current_year and existing_year:
                        # Existing has year, current doesn't - keep existing
                        duplicate_groups.append({
                            'kept': f"{existing_row['title'][:50]}...",
                            'removed': f"{current_row['title'][:50]}...",
                            'reason': 'Missing publication year data'
                        })
                    else:
                        # Neither has year data, fall back to completeness
                        current_score = calculate_completeness_score(current_row)
                        existing_score = calculate_completeness_score(existing_row)
                        
                        if current_score > existing_score:
                            duplicate_groups.append({
                                'kept': f"{current_row['title'][:50]}...",
                                'removed': f"{existing_row['title'][:50]}...",
                                'reason': f'More complete record (score: {current_score:.1f} vs {existing_score:.1f})'
                            })
                            citations_to_keep[j] = current_row
                        else:
                            duplicate_groups.append({
                                'kept': f"{existing_row['title'][:50]}...",
                                'removed': f"{current_row['title'][:50]}...",
                                'reason': f'Less complete record (score: {current_score:.1f} vs {existing_score:.1f})'
                            })
                    
                    removal_strategy = "Year-based prioritization with completeness fallback"
                else:
                    # Strategy 2: No year data - use completeness or upload order
                    current_score = calculate_completeness_score(current_row)
                    existing_score = calculate_completeness_score(existing_row)
                    
                    if abs(current_score - existing_score) > 5:  # Significant difference
                        if current_score > existing_score:
                            duplicate_groups.append({
                                'kept': f"{current_row['title'][:50]}...",
                                'removed': f"{existing_row['title'][:50]}...",
                                'reason': f'More complete record (score: {current_score:.1f} vs {existing_score:.1f})'
                            })
                            citations_to_keep[j] = current_row
                        else:
                            duplicate_groups.append({
                                'kept': f"{existing_row['title'][:50]}...",
                                'removed': f"{current_row['title'][:50]}...",
                                'reason': f'Less complete record (score: {current_score:.1f} vs {existing_score:.1f})'
                            })
                    else:
                        # Similar completeness - keep first occurrence (upload order)
                        duplicate_groups.append({
                            'kept': f"{existing_row['title'][:50]}...",
                            'removed': f"{current_row['title'][:50]}...",
                            'reason': 'Upload order priority (similar completeness)'
                        })
                    
                    removal_strategy = "Completeness-based with upload order fallback"
                
                break
        
        if not is_duplicate:
            citations_to_keep.append(current_row)
    
    # Create Citation objects
    new_citations = []
    for row in citations_to_keep:
        citation = Citation(
            title=str(row['title']),
            abstract=str(row['abstract']),
            project_id=project_id,
            iteration=current_iteration
        )
        new_citations.append(citation)
    
    return {
        'error': None,
        'citations': new_citations,
        'duplicates_removed': len(duplicate_groups),
        'removal_strategy': removal_strategy,
        'duplicate_details': duplicate_groups[:10]  # Limit to first 10 for response size
    }
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "X-User-Id"]
}})

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

    try:
        # Convert user_id to integer and strictly filter by user_id
        user_id_int = int(user_id)
        projects = Project.query.filter_by(user_id=user_id_int).all()
        app.logger.info(f"Found {len(projects)} projects for user {user_id_int}")
        
        return jsonify({
            "projects": [{
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at,
                "current_iteration": p.current_iteration
            } for p in projects]
        })
    except Exception as e:
        return jsonify({"error": "Failed to fetch projects"}), 500


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

        user_id_int = int(user_id)
        project = Project(name=data['name'],
                        user_id=user_id_int,
                        keywords={
                            "include": [],
                            "exclude": []
                        })
        db.session.add(project)
        db.session.commit()
        db.session.refresh(project)
        
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

                # Process duplicates and create citations
                result = process_duplicates_and_create_citations(df, project_id, project.current_iteration)
                
                if result['error']:
                    return jsonify({"error": result['error']}), 400

                db.session.bulk_save_objects(result['citations'])
                db.session.commit()
                
                response_data = {
                    "message": f"Added {len(result['citations'])} citations"
                }
                
                if result['duplicates_removed'] > 0:
                    response_data["duplicates_info"] = {
                        "duplicates_removed": result['duplicates_removed'],
                        "removal_strategy": result['removal_strategy'],
                        "details": result['duplicate_details']
                    }
                
                return jsonify(response_data), 201

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


@app.route('/api/projects/<int:project_id>/citations/<int:citation_id>/relevance',
           methods=['PUT'])
def update_citation(project_id, citation_id):
    try:
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
            # Allow null/None value to unlabel the citation
            citation.is_relevant = data['is_relevant'] if data['is_relevant'] in [True, False] else None
            if citation.is_relevant is not None:
                citation.iteration = project.current_iteration
            db.session.commit()
            status = "unlabeled" if citation.is_relevant is None else str(citation.is_relevant)
            app.logger.info(f"Updated citation {citation_id} relevance to {status} for iteration {citation.iteration}")

        return jsonify({
            "citation": {
                "id": citation.id,
                "title": citation.title,
                "abstract": citation.abstract,
                "is_relevant": citation.is_relevant,
                "iteration": citation.iteration
            }
        })
    except Exception as e:
        app.logger.error(f"Error updating citation: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Failed to update citation"}), 500


@app.route('/api/projects/<int:project_id>/train', methods=['POST'])
def train_model(project_id):
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    project = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404

    # Process citations based on exclude keywords
    citations = Citation.query.filter_by(project_id=project_id).all()
    excluded_citations = []
    exclude_reasons = {}

    for citation in citations:
        text = f"{citation.title} {citation.abstract}".lower()
        for exclude_kw in project.keywords.get('exclude', []):
            word = exclude_kw['word'].lower()
            frequency = exclude_kw.get('frequency', 1)
            occurrences = text.count(word)
            
            if occurrences >= frequency:
                excluded_citations.append(citation.id)
                if word not in exclude_reasons:
                    exclude_reasons[word] = {'frequency': frequency, 'count': 0}
                exclude_reasons[word]['count'] += 1
                break

    # Filter out excluded citations
    Citation.query.filter(Citation.id.in_(excluded_citations)).update(
        {Citation.is_relevant: False}, synchronize_session=False
    )
    db.session.commit()

    review_system = LiteratureReviewSystem(project_id)
    result = review_system.train_iteration(project.current_iteration)

    if 'error' in result:
        return jsonify(result), 400

    # Add exclusion metadata to result
    result['excluded_citations'] = {
        'total': len(excluded_citations),
        'reasons': [
            {'keyword': k, 'frequency': v['frequency'], 'citations_excluded': v['count']}
            for k, v in exclude_reasons.items()
        ]
    }

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
    vectorizer = TfidfVectorizer(max_features=70, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get feature names and their scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    # Get top 50 keywords by TF-IDF score but only return the words
    top_indices = scores.argsort()[-50:][::-1]  
    suggested_keywords = [{"word": feature_names[i]} for i in top_indices]

    return jsonify({
        "suggested_keywords": suggested_keywords,
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
    if not isinstance(data, dict) or not all(k in data for k in ['include', 'exclude']):
        return jsonify({"error": "Invalid keywords format"}), 400

    # Validate exclude keywords format
    if not all(isinstance(item, dict) and 'word' in item and 'frequency' in item 
              for item in data['exclude']):
        return jsonify({"error": "Invalid exclude keywords format"}), 400

    # Set default frequency of 1 if not specified
    for item in data['exclude']:
        if not isinstance(item['frequency'], int) or item['frequency'] < 1:
            item['frequency'] = 1

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

    # Create list of data and sort by relevance score
    data_rows = []
    for citation, prediction in zip(citations, predictions):
        relevance_score = prediction.get('relevance_probability', 0) if 'error' not in prediction else 0
        data_rows.append({
            'title': citation.title,
            'abstract': citation.abstract,
            'is_relevant': 'Yes' if citation.is_relevant else 'No' if citation.is_relevant is not None else 'Unclassified',
            'iteration': citation.iteration,
            'relevance_score': relevance_score
        })
    
    # Sort by relevance score in descending order
    data_rows.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Write sorted data
    for row, data in enumerate(data_rows, start=2):
        worksheet.cell(row=row, column=1, value=data['title'])
        worksheet.cell(row=row, column=2, value=data['abstract'])
        worksheet.cell(row=row, column=3, value=data['is_relevant'])
        worksheet.cell(row=row, column=4, value=data['iteration'])
        worksheet.cell(row=row, column=5, value=data['relevance_score'])

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


@app.route('/api/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    try:
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401

        project = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if not project:
            return jsonify({"error": "Project not found"}), 404

        Citation.query.filter_by(project_id=project_id).delete()
        db.session.delete(project)
        db.session.commit()
        
        return '', 204

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to delete project"}), 500

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