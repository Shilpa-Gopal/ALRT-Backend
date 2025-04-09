import pandas as pd
import numpy as np
from xgboost import XGBClassifier, DMatrix, train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.cluster import KMeans
from scipy import sparse
import logging
from typing import Dict, Tuple
import os

class LiteratureReviewSystem:
    def __init__(self, max_features: int = 30, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.samples_per_class = 5

        self.keywords = {
            'include_keywords': [
                'trial', 'clinical trial', 'rct', 'cct', 'randomized', 'randomised',
                'controlled trial', 'crossover', 'parallel group',
                'double blind', 'single blind', 'placebo controlled',
                'case-control study', 'cohort study', 'observational study',
                'ctdna', 'cancer', 'hpv', 'granulosa', 'compared with', 'control groups'
            ],
            'exclude_keywords': [
                'cells', 'in vitro', 'case report', 'single arm',
                'systematic review', 'meta-analysis', 'literature review',
                'mouse', 'mice', 'rat', 'rats', 'rodent', 'rodents',
                'fish', 'porcine', 'animal', 'animals', 'murine',
                'rabbit', 'rabbits', 'canine'
            ]
        }

        self.vectorizer = TfidfVectorizer(
            vocabulary=self.keywords['include_keywords'] + self.keywords['exclude_keywords'],
            ngram_range=(1, 2),
            stop_words='english'
        )

        self.model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }

        self._setup_logging()

    def _setup_logging(self):
        os.makedirs('data/sample', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('literature_review_xgboost.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_features(self, data: pd.DataFrame, is_training: bool = False) -> sparse.csr_matrix:
        combined_text = data.apply(
            lambda x: f"{str(x['title'] or '')} {str(x['abstract'] or '')}",
            axis=1
        )

        keyword_features = self.vectorizer.fit_transform(combined_text) if is_training else self.vectorizer.transform(combined_text)

        # Add abstract word count as an additional feature
        data = data.copy()
        data['abstract_word_count'] = data['abstract'].apply(lambda x: len(str(x).split()))
        additional_features = data[['abstract_word_count']].values

        return sparse.hstack([keyword_features, sparse.csr_matrix(additional_features)])

    def calculate_metrics(self, y_true, y_pred_probs, iteration: int):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        y_pred_adjusted = (y_pred_probs >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_adjusted).ravel()

        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': precision_score(y_true, y_pred_adjusted),
            'recall': recall_score(y_true, y_pred_adjusted),
            'f1': f1_score(y_true, y_pred_adjusted),
            'accuracy': accuracy_score(y_true, y_pred_adjusted)
        }

        self.logger.info(f"\nIteration {iteration} Metrics:")
        self.logger.info("-" * 50)
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"True Positives: {tp}")
        self.logger.info(f"True Negatives: {tn}")
        self.logger.info(f"False Positives: {fp}")
        self.logger.info(f"False Negatives: {fn}")
        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        return metrics, optimal_threshold

    def select_samples_for_labeling(self, remaining_data: pd.DataFrame, method: str = 'uncertainty') -> pd.DataFrame:
        remaining_data = remaining_data.copy()  # Ensure we are working with a copy

        if method == 'uncertainty':
            # Sort by uncertainty (closest to 0.5)
            remaining_data['uncertainty'] = np.abs(remaining_data['predicted_prob'] - 0.5)
            sorted_data = remaining_data.sort_values(by='uncertainty', ascending=True)
            return sorted_data.head(self.samples_per_class * 2)

        elif method == 'entropy':
            # Calculate entropy
            remaining_data['entropy'] = - (
                remaining_data['predicted_prob'] * np.log2(remaining_data['predicted_prob'] + 1e-8) +
                (1 - remaining_data['predicted_prob']) * np.log2(1 - remaining_data['predicted_prob'] + 1e-8)
            )
            sorted_data = remaining_data.sort_values(by='entropy', ascending=False)
            return sorted_data.head(self.samples_per_class * 2)

        elif method == 'clustering':
            # Clustering-based sampling
            X_remaining = self.prepare_features(remaining_data)
            kmeans = KMeans(n_clusters=10, random_state=42).fit(X_remaining)
            remaining_data['cluster'] = kmeans.labels_
            return remaining_data.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)

        else:
            raise ValueError("Unsupported sampling method")

    def iterative_training(self, labeled_data: pd.DataFrame, n_iterations: int = 5, sampling_method: str = 'uncertainty') -> Dict:
        history = {
            'iterations': [], 
            'metrics': [], 
            'training_samples': [],
            'citations_checked': []
        }

        cumulative_train_indices = set()
        booster = None

        for iteration in range(n_iterations):
            self.logger.info(f"\n{'='*20} Iteration {iteration + 1} {'='*20}")

            self.samples_per_class = min(10 + iteration * 5, 50)

            if iteration == 0:
                relevant = labeled_data[labeled_data['relevant'] == 1].head(self.samples_per_class)
                irrelevant = labeled_data[labeled_data['relevant'] == 0].head(self.samples_per_class)
                new_samples = pd.concat([relevant, irrelevant])
                citations_checked = self.samples_per_class * 2
            else:
                remaining_data = labeled_data[~labeled_data.index.isin(cumulative_train_indices)].copy()
                X_remaining = self.prepare_features(remaining_data)

                if booster is not None:
                    dremaining = DMatrix(X_remaining)
                    remaining_data['predicted_prob'] = booster.predict(dremaining)

                # Select samples based on the chosen method
                new_samples = self.select_samples_for_labeling(remaining_data, method=sampling_method)
                citations_checked = len(new_samples)

            cumulative_train_indices.update(new_samples.index)

            train_data = labeled_data.loc[list(cumulative_train_indices)]
            test_data = labeled_data[~labeled_data.index.isin(cumulative_train_indices)]

            X_train = self.prepare_features(train_data, is_training=True)
            y_train = train_data['relevant'].values

            X_test = self.prepare_features(test_data)
            y_test = test_data['relevant'].values

            dtrain = DMatrix(X_train, label=y_train)
            dtest = DMatrix(X_test, label=y_test)

            self.model_params['scale_pos_weight'] = len(train_data[train_data['relevant'] == 0]) / len(train_data[train_data['relevant'] == 1])

            evals = [(dtrain, 'train'), (dtest, 'eval')]

            booster = train(
                self.model_params,
                dtrain,
                num_boost_round=100,
                early_stopping_rounds=10,
                evals=evals,
                verbose_eval=False
            )

            y_test_probs = booster.predict(dtest)
            metrics, threshold = self.calculate_metrics(y_test, y_test_probs, iteration + 1)

            history['iterations'].append(iteration + 1)
            history['metrics'].append(metrics)
            history['training_samples'].append(len(train_data))
            history['citations_checked'].append(citations_checked)

        return history


def main():
    try:
        labeled_data = pd.read_excel("AI-Ranking-tools-Literature-review-/data/processed/citations_with_llm_scores.xlsx")
        print(f"Loaded {len(labeled_data)} citations")

        system = LiteratureReviewSystem(max_features=20)

        print("\nRunning training with XGBoost model...")
        history = system.iterative_training(labeled_data=labeled_data, n_iterations=10, sampling_method='uncertainty')

        results = []
        for i in range(len(history['iterations'])):
            results.append({
                'iteration': history['iterations'][i],
                'training_samples': history['training_samples'][i],
                'citations_checked': history['citations_checked'][i],
                'precision': history['metrics'][i]['precision'],
                'recall': history['metrics'][i]['recall'],
                'f1': history['metrics'][i]['f1'],
                'accuracy': history['metrics'][i]['accuracy']
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv('xgboost_training_results.csv', index=False)
        print("\nResults saved to xgboost_training_results.csv")
        print("\nResults:")
        print(results_df)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
