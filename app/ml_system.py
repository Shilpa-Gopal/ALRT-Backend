
from .models import Citation, Project
from app import db
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
import logging
import os

class LiteratureReviewSystem:
    def __init__(self, project_id: int, max_features: int = 30):
        self.project_id = project_id
        self.max_features = max_features
        self.setup_logging()
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
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
                'iteration': citation.iteration,
                'word_count': len(str(citation.abstract).split())
            })
        return pd.DataFrame(data)

    def prepare_features(self, data):
        # Combine title and abstract
        text = data.apply(lambda x: f"{str(x['title'])} {str(x['abstract'])}", axis=1)
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(text)
        
        # Additional numerical features
        word_counts = data['word_count'].values.reshape(-1, 1)
        
        # Combine features
        return np.hstack([tfidf_features.toarray(), word_counts])

    def predict_relevance(self, citations):
        try:
            # Convert citations to DataFrame
            new_data = pd.DataFrame(citations)
            new_data['word_count'] = new_data['abstract'].apply(lambda x: len(str(x).split()))
            
            # Get existing data and prepare model
            data = self.get_project_data()
            labeled_data = data[data['is_relevant'].notna()]
            
            if len(labeled_data) < 10:
                return [{"error": "Not enough labeled data for prediction"}] * len(citations)
            
            # Prepare features and train model
            X = self.prepare_features(labeled_data)
            y = labeled_data['is_relevant'].astype(int)
            
            model = XGBClassifier(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
            model.fit(X, y)
            
            # Predict on new data
            X_new = self.prepare_features(new_data)
            predictions = model.predict_proba(X_new)
            
            # Calculate optimal threshold
            self.optimal_threshold = self.find_optimal_threshold(y, model.predict_proba(X)[:, 1])
            
            # Return predictions
            return [{
                "citation_index": i,
                "relevance_probability": float(pred[1]),
                "is_relevant": bool(pred[1] > self.optimal_threshold)
            } for i, pred in enumerate(predictions)]
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return [{"error": "Prediction failed"}] * len(citations)

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
        data = self.get_project_data()
        labeled_data = data[data['is_relevant'].notna()]
        
        if len(labeled_data) < 10:
            return {'error': 'Not enough labeled data'}
            
        X = self.prepare_features(labeled_data)
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
        
        return {
            'metrics': metrics,
            'samples_checked': len(y)
        }
