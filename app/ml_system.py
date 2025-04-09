
from .models import Citation, Project
from app import db
import pandas as pd
import numpy as np
from xgboost import DMatrix, train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.cluster import KMeans
from scipy import sparse
import logging
import os

class LiteratureReviewSystem:
    def __init__(self, project_id: int, max_features: int = 30):
        self.project_id = project_id
        self.max_features = max_features
        self.setup_logging()
        
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

    def get_project_data(self) -> pd.DataFrame:
        citations = Citation.query.filter_by(project_id=self.project_id).all()
        data = []
        for citation in citations:
            data.append({
                'id': citation.id,
                'title': citation.title,
                'abstract': citation.abstract,
                'is_relevant': citation.is_relevant,
                'iteration': citation.iteration
            })
        return pd.DataFrame(data)
        
    def predict_relevance(self, citations):
        try:
            # Convert citations to DataFrame
            new_data = pd.DataFrame(citations)
            
            # Get existing data and prepare model
            data = self.get_project_data()
            labeled_data = data[data['is_relevant'].notna()]
            
            if len(labeled_data) < 10:
                return [{"error": "Not enough labeled data for prediction"}] * len(citations)
                
            # Train model on existing data
            system = LiteratureReviewSystem(max_features=self.max_features)
            history = system.iterative_training(labeled_data, n_iterations=1)
            
            # Prepare features for new citations
            X_new = system.prepare_features(new_data)
            predictions = system.model.predict_proba(X_new)
            
            # Return predictions with confidence scores
            return [{
                "citation_index": i,
                "relevance_probability": float(pred[1]),
                "is_relevant": bool(pred[1] > system.optimal_threshold)
            } for i, pred in enumerate(predictions)]
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return [{"error": "Prediction failed"}] * len(citations)

    def train_iteration(self, iteration: int = 0):
        data = self.get_project_data()
        labeled_data = data[data['is_relevant'].notna()]
        
        if len(labeled_data) < 10:
            return {'error': 'Not enough labeled data'}
            
        system = LiteratureReviewSystem(max_features=self.max_features)
        history = system.iterative_training(labeled_data, n_iterations=1)
        
        return {
            'metrics': history['metrics'][0],
            'samples_checked': history['citations_checked'][0]
        }
