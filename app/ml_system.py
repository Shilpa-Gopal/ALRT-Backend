import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from .models import Citation, db
import logging
import os

class LiteratureReviewSystem:
    def __init__(self, project_id, max_features: int = 1000):
        self.project_id = project_id
        self.max_features = max_features
        self.setup_logging()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
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
        citations = Citation.query.filter_by(project_id=self.project_id).all()
        return pd.DataFrame([{
            'title': c.title,
            'abstract': c.abstract,
            'is_relevant': c.is_relevant
        } for c in citations])

    def prepare_features(self, data):
        text_data = data['title'] + ' ' + data['abstract']
        features = self.vectorizer.fit_transform(text_data)
        return features

    def predict_relevance(self, citations):
        try:
            new_data = pd.DataFrame(citations)
            data = self.get_project_data()
            labeled_data = data[data['is_relevant'].notna()]

            if len(labeled_data) < 10:
                return [{"error": "Not enough labeled data for prediction"}] * len(citations)

            X = self.prepare_features(labeled_data)
            y = labeled_data['is_relevant'].astype(int)

            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
            model.fit(X, y)

            X_new = self.prepare_features(new_data)
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
        data = self.get_project_data()
        labeled_data = data[data['is_relevant'].notna()]
        
        # Validate citation counts for current iteration
        current_iter_data = labeled_data[labeled_data['iteration'] == iteration]
        relevant_count = sum(current_iter_data['is_relevant'] == True)
        irrelevant_count = sum(current_iter_data['is_relevant'] == False)
        
        if relevant_count != 5 or irrelevant_count != 5:
            return {'error': 'Each iteration requires exactly 5 relevant and 5 irrelevant citations'}

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