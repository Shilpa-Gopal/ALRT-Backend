
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
