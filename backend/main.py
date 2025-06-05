from flask import Flask, jsonify, request, abort, send_file
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
import xlsxwriter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from app import create_app, db
from app.models import User, Project, Citation
from app.ml_system import LiteratureReviewSystem
import hashlib
import re

app = create_app()

def normalize_text(text):
    """Normalize text for better matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove extra spaces, punctuation, and normalize
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_text_hash(title, abstract, authors=None):
    """Create a hash for quick duplicate detection"""
    combined = f"{normalize_text(title)} {normalize_text(abstract)}"
    if authors:
        combined += f" {normalize_text(authors)}"
    return hashlib.md5(combined.encode()).hexdigest()

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
        score += min(abstract_len / 10, 50)
    
    return score

def process_duplicates_and_create_citations(df, project_id, current_iteration):
    """Optimized duplicate processing using vectorization and hashing"""
    
    app.logger.info(f"Processing {len(df)} citations for duplicates")
    
    # Normalize column names
    df_normalized = df.copy()
    df_normalized.columns = df_normalized.columns.str.lower()
    
    # Check required columns
    if 'title' not in df_normalized.columns or 'abstract' not in df_normalized.columns:
        return {
            'error': f"File must contain 'title' and 'abstract' columns. Found: {list(df.columns)}",
            'citations': [],
            'duplicates_removed': 0,
            'removal_strategy': '',
            'duplicate_details': []
        }
    
    # Find optional columns
    year_col = find_year_column(df)
    authors_col = None
    for col in df.columns:
        if 'author' in col.lower():
            authors_col = col
            break
    
    # Step 1: Quick hash-based duplicate detection for exact matches
    df['text_hash'] = df.apply(lambda row: create_text_hash(
        row['title'], 
        row['abstract'], 
        row.get(authors_col) if authors_col else None
    ), axis=1)
    
    # Step 2: Group by hash to find exact duplicates
    hash_groups = df.groupby('text_hash')
    
    # Step 3: For groups with multiple items, apply selection strategy
    final_citations = []
    duplicate_details = []
    removal_strategy = "Hash-based with TF-IDF similarity fallback"
    
    for hash_val, group in hash_groups:
        if len(group) == 1:
            # No duplicates
            final_citations.append(group.iloc[0])
        else:
            # Handle duplicates within this group
            app.logger.info(f"Found {len(group)} potential duplicates")
            
            # Select best citation from this group
            best_citation = select_best_citation(group, year_col, duplicate_details)
            final_citations.append(best_citation)
    
    # Step 4: For remaining potential near-duplicates, use TF-IDF
    if len(final_citations) > 1:
        final_citations = remove_similar_citations(final_citations, authors_col, duplicate_details)
    
    # Create Citation objects
    new_citations = []
    for _, row in enumerate(final_citations):
        citation = Citation(
            title=str(row['title']),
            abstract=str(row['abstract']),
            project_id=project_id,
            iteration=current_iteration
        )
        new_citations.append(citation)
    
    duplicates_removed = len(df) - len(new_citations)
    app.logger.info(f"Removed {duplicates_removed} duplicates, keeping {len(new_citations)} citations")
    
    return {
        'error': None,
        'citations': new_citations,
        'duplicates_removed': duplicates_removed,
        'removal_strategy': removal_strategy,
        'duplicate_details': duplicate_details[:10]
    }

def select_best_citation(group, year_col, duplicate_details):
    """Select the best citation from a group of duplicates"""
    if len(group) == 1:
        return group.iloc[0]
    
    # Strategy 1: Use year if available
    if year_col:
        group['year_extracted'] = group[year_col].apply(extract_year_from_value)
        valid_years = group[group['year_extracted'].notna()]
        
        if len(valid_years) > 0:
            # Keep the most recent
            best = valid_years.loc[valid_years['year_extracted'].idxmax()]
            
            # Log the duplicate removal
            for _, other in group.iterrows():
                if other.name != best.name:
                    duplicate_details.append({
                        'kept': f"{best['title'][:50]}...",
                        'removed': f"{other['title'][:50]}...",
                        'reason': f'Year-based selection'
                    })
            
            return best
    
    # Strategy 2: Use completeness score
    group['completeness'] = group.apply(calculate_completeness_score, axis=1)
    best = group.loc[group['completeness'].idxmax()]
    
    # Log the duplicate removal
    for _, other in group.iterrows():
        if other.name != best.name:
            duplicate_details.append({
                'kept': f"{best['title'][:50]}...",
                'removed': f"{other['title'][:50]}...",
                'reason': f'Completeness-based selection'
            })
    
    return best

def remove_similar_citations(citations_list, authors_col, duplicate_details):
    """Use TF-IDF to find and remove similar citations"""
    if len(citations_list) < 2:
        return citations_list
    
    # Prepare text for TF-IDF
    texts = []
    for citation in citations_list:
        text = f"{normalize_text(citation['title'])} {normalize_text(citation['abstract'])}"
        if authors_col and pd.notna(citation.get(authors_col)):
            text += f" {normalize_text(citation[authors_col])}"
        texts.append(text)
    
    # Calculate TF-IDF similarity only if we have enough text variation
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar pairs (above 70% similarity)
        to_remove = set()
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > 0.7:
                    # Keep the one with higher completeness
                    citation_i = citations_list[i]
                    citation_j = citations_list[j]
                    
                    score_i = calculate_completeness_score(citation_i)
                    score_j = calculate_completeness_score(citation_j)
                    
                    if score_i >= score_j:
                        to_remove.add(j)
                        duplicate_details.append({
                            'kept': f"{citation_i['title'][:50]}...",
                            'removed': f"{citation_j['title'][:50]}...",
                            'reason': f'TF-IDF similarity: {similarity_matrix[i][j]:.2f}'
                        })
                    else:
                        to_remove.add(i)
                        duplicate_details.append({
                            'kept': f"{citation_j['title'][:50]}...",
                            'removed': f"{citation_i['title'][:50]}...",
                            'reason': f'TF-IDF similarity: {similarity_matrix[i][j]:.2f}'
                        })
        
        # Return citations not in removal set
        return [citation for i, citation in enumerate(citations_list) if i not in to_remove]
        
    except Exception as e:
        app.logger.warning(f"TF-IDF similarity calculation failed: {e}")
        return citations_list
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
                    
                    # Check file size limits for performance
                    if len(df) > 10000:
                        return jsonify({
                            "error": "File too large",
                            "details": f"File contains {len(df)} rows. Maximum supported is 10,000 rows for optimal performance."
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

                # Create citations without duplicate processing
                if 'title' not in df.columns or 'abstract' not in df.columns:
                    return jsonify({
                        "error": "File must contain 'title' and 'abstract' columns",
                        "found_columns": list(df.columns)
                    }), 400

                new_citations = []
                for _, row in df.iterrows():
                    citation = Citation(
                        title=str(row['title']),
                        abstract=str(row['abstract']),
                        project_id=project_id,
                        iteration=project.current_iteration
                    )
                    new_citations.append(citation)

                db.session.bulk_save_objects(new_citations)
                db.session.commit()
                
                return jsonify({
                    "message": f"Added {len(new_citations)} citations",
                    "total_citations": len(new_citations),
                    "note": "Use 'Detect and Remove Duplicates' button in keywords section to remove any duplicates"
                }), 201

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


@app.route('/api/projects/<int:project_id>/remove-duplicates', methods=['POST'])
def remove_duplicates(project_id):
    try:
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401

        project = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if not project:
            return jsonify({"error": "Project not found"}), 404

        # Get all citations for this project
        citations = Citation.query.filter_by(project_id=project_id).all()
        
        if not citations:
            return jsonify({"error": "No citations found in this project"}), 404

        app.logger.info(f"Starting duplicate removal for project {project_id} with {len(citations)} citations")

        # Convert citations to DataFrame format for processing
        citation_data = []
        for citation in citations:
            citation_data.append({
                'id': citation.id,
                'title': citation.title,
                'abstract': citation.abstract,
                'is_relevant': citation.is_relevant,
                'iteration': citation.iteration
            })

        df = pd.DataFrame(citation_data)
        
        # Process duplicates using existing logic
        result = process_duplicates_and_create_citations(df, project_id, project.current_iteration)
        
        if result['error']:
            return jsonify({"error": result['error']}), 400

        # If duplicates were found, update the database
        if result['duplicates_removed'] > 0:
            # Get IDs of citations to keep
            citations_to_keep = [c.title for c in result['citations']]
            
            # Delete duplicate citations from database
            # Keep the first occurrence of each unique citation
            kept_citation_ids = []
            for citation in citations:
                citation_key = f"{citation.title.lower().strip()} {citation.abstract.lower().strip()}"
                
                # Check if this citation should be kept
                should_keep = False
                for kept_citation in result['citations']:
                    kept_key = f"{kept_citation['title'].lower().strip()} {kept_citation['abstract'].lower().strip()}"
                    if citation_key == kept_key and citation.id not in kept_citation_ids:
                        kept_citation_ids.append(citation.id)
                        should_keep = True
                        break
                
                if not should_keep:
                    db.session.delete(citation)
            
            db.session.commit()
            app.logger.info(f"Removed {result['duplicates_removed']} duplicate citations from project {project_id}")

        return jsonify({
            "message": f"Duplicate removal completed",
            "duplicates_removed": result['duplicates_removed'],
            "remaining_citations": len(result['citations']),
            "removal_strategy": result['removal_strategy'],
            "duplicate_details": result['duplicate_details'][:5]  # Show first 5 examples
        }), 200

    except Exception as e:
        app.logger.error(f"Error removing duplicates: {str(e)}")
        db.session.rollback()
        return jsonify({
            "error": "Failed to remove duplicates",
            "details": str(e)
        }), 500