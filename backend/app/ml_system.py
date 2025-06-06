import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from scipy import sparse
from .models import Citation, db
import logging
import os

class LiteratureReviewSystem:
    def __init__(self, project_id, max_features: int = 1000):
        self.project_id = project_id
        self.max_features = max_features
        self.setup_logging()
        self.vectorizer = TfidfVectorizer(max_features=max_features) #Updated max_features here
        self.optimal_threshold = 0.5
        self.max_iterations = 10
        self.labeled_citations = set()

    def setup_logging(self):
        os.makedirs('data/logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'data/logs/project_{self.project_id}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'project_{self.project_id}')

    def get_project_data(self):
        # Only get non-duplicate citations for training
        citations = Citation.query.filter_by(
            project_id=self.project_id,
            is_duplicate=False
        ).all()
        return pd.DataFrame([{
            'title': c.title,
            'abstract': c.abstract,
            'is_relevant': c.is_relevant,
            'iteration': c.iteration
        } for c in citations])

    def prepare_features(self, data, project=None):
        text_data = data['title'] + ' ' + data['abstract']
        
        # Base TF-IDF features
        base_features = self.vectorizer.fit_transform(text_data)
        
        # Add include keyword features if project keywords are available
        if project and project.keywords and project.keywords.get('include'):
            include_keywords = [kw.lower() for kw in project.keywords.get('include', [])]
            
            # Create include keyword features
            include_features = []
            for _, row in data.iterrows():
                text = f"{row['title']} {row['abstract']}".lower()
                keyword_scores = []
                
                for keyword in include_keywords:
                    # Count occurrences and normalize by text length
                    occurrences = text.count(keyword)
                    normalized_score = occurrences / max(len(text.split()), 1)
                    keyword_scores.append(normalized_score)
                
                include_features.append(keyword_scores)
            
            # Convert to sparse matrix and combine with base features
            include_matrix = sparse.csr_matrix(include_features)
            features = sparse.hstack([base_features, include_matrix])
            
            self.logger.info(f"Enhanced features with {len(include_keywords)} include keywords")
            return features
        
        return base_features

    def predict_relevance(self, citations):
        try:
            from .models import Project
            
            new_data = pd.DataFrame(citations)
            data = self.get_project_data()
            labeled_data = data[data['is_relevant'].notna()]

            if len(labeled_data) < 10:
                return [{"error": "Not enough labeled data for prediction"}] * len(citations)

            # Get project for keyword enhancement
            project = Project.query.get(self.project_id)

            X = self.prepare_features(labeled_data, project)
            y = labeled_data['is_relevant'].astype(int)

            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
            model.fit(X, y)

            X_new = self.prepare_features(new_data, project)
            predictions = model.predict_proba(X_new)

            return [{
                "citation_index": i,
                "relevance_probability": float(pred[1]),
                "is_relevant": bool(pred[1] > self.optimal_threshold)
            } for i, pred in enumerate(predictions)]

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return [{"error": str(e)}] * len(citations)

    def calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    def find_optimal_threshold(self, y_true, y_pred_proba):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    def train_iteration(self, iteration: int = 0):
        try:
            from .models import Project
            
            data = self.get_project_data()
            labeled_data = data[data['is_relevant'].notna()]

            if len(labeled_data) < 10:
                return {'error': 'Need at least 10 labeled citations to train the model'}

            # Count relevant and irrelevant citations
            relevant_count = sum(labeled_data['is_relevant'] == True)
            irrelevant_count = sum(labeled_data['is_relevant'] == False)

            if relevant_count < 5 or irrelevant_count < 5:
                return {'error': f'Need at least 5 relevant and 5 irrelevant citations. Have {relevant_count} relevant and {irrelevant_count} irrelevant.'}

            # Get project for keyword enhancement
            project = Project.query.get(self.project_id)
            
            # Log include keyword usage
            include_keywords = project.keywords.get('include', []) if project.keywords else []
            self.logger.info(f"Training with {len(labeled_data)} labeled citations ({relevant_count} relevant, {irrelevant_count} irrelevant)")
            if include_keywords:
                self.logger.info(f"Using {len(include_keywords)} include keywords for feature enhancement: {include_keywords}")

            X = self.prepare_features(labeled_data, project)
            y = labeled_data['is_relevant'].astype(int)

            model = XGBClassifier(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )

            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = self.calculate_metrics(y, y_pred)

            self.logger.info(f"Training completed successfully. Metrics: {metrics}")

            return {
                'metrics': metrics,
                'samples_checked': len(y),
                'relevant_count': relevant_count,
                'irrelevant_count': irrelevant_count,
                'include_keywords_used': len(include_keywords)
            }

        except Exception as e:
            self.logger.error(f"Training iteration error: {str(e)}")
            return {'error': f'Training failed: {str(e)}'}