import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import implicit
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender(ABC):
    """Abstract Base Class for all recommendation models."""
    
    def __init__(self, **kwargs):
        self.model = None
    
    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix) -> None:
        pass
    
    @abstractmethod
    def recommend(self, target_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        pass

    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Universal wrapper to ensure global compatibility with the evaluation pipeline.
        Automatically routes the evaluation call to the standard recommend method.

        Args:
            user_id (int): The index of the target user.
            n_recommendations (int, optional): Number of items to return. Defaults to 10.

        Returns:
            List[Tuple[int, float]]: A list of tuples containing (item_index, prediction_score).
        """
        return self.recommend(target_id=user_id, n_recommendations=n_recommendations)


class ItemBasedKNN(BaseRecommender):
    """Item-Based Collaborative Filtering using K-Nearest Neighbors."""
    
    def __init__(self, n_neighbors=20, metric='cosine', algorithm='brute'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm=algorithm,
            n_jobs=-1
        )
        self.item_matrix = None
    
    def fit(self, interaction_matrix):
        print(f"[INFO] Fitting Item-Based KNN with metric='{self.metric}'...")
        self.item_matrix = interaction_matrix.T
        self.model.fit(self.item_matrix)
        print("[INFO] KNN Model fitting complete. ")
    
    def recommend(self, item_id, n_recommendations=10):
        if self.item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        target_item_vector = self.item_matrix[item_id]
        distances, indices = self.model.kneighbors(
            target_item_vector,
            n_neighbors=n_recommendations + 1
        )
        
        distances = distances.flatten()[1:]
        indices = indices.flatten()[1:]
        
        return list(zip(indices, distances))
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        if self.item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        user_interacted_items = self.item_matrix[:, user_id].nonzero()[0]
        
        if len(user_interacted_items) == 0:
            return []
        
        candidate_items = {}
        
        for item_id in user_interacted_items:
            recs = self.recommend(item_id, n_recommendations=5)
            
            for rec_id, distance in recs:
                if rec_id not in user_interacted_items:
                    sim_score = 1 - distance
                    if rec_id in candidate_items:
                        candidate_items[rec_id] += sim_score
                    else:
                        candidate_items[rec_id] = sim_score
        
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:n_recommendations]


class ALSRecommender(BaseRecommender):
    """Alternating Least Squares (ALS) Recommender using 'implicit' library."""
    
    def __init__(self, factors=64, regularization=0.1, iterations=20, random_state=42):
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )
        self.user_item_matrix = None
    
    def fit(self, interaction_matrix):
        print(f"[INFO] Fitting ALS Model with {self.factors} latent factors...")
        self.user_item_matrix = interaction_matrix
        self.model.fit(self.user_item_matrix)
        print("[INFO] ALS Model fitting complete. ")
    
    def recommend(self, item_id, n_recommendations=10):
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        item_ids, scores = self.model.similar_items(item_id, N=n_recommendations + 1)
        
        recommendations = []
        for idx, score in zip(item_ids, scores):
            if idx != item_id:
                recommendations.append((idx, score))
        
        return recommendations[:n_recommendations]
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        item_ids, scores = self.model.recommend(
            userid=user_id,
            user_items=self.user_item_matrix[user_id],
            N=n_recommendations
        )
        return list(zip(item_ids, scores))


class BPRRecommender(BaseRecommender):
    """Bayesian Personalized Ranking (BPR) Recommender using 'implicit' library."""
    
    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01, iterations=100, random_state=42):
        super().__init__()
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )
        self.user_item_matrix = None
    
    def fit(self, interaction_matrix):
        print(f"[INFO] Fitting BPR Model with {self.factors} latent factors...")
        self.user_item_matrix = interaction_matrix
        self.model.fit(self.user_item_matrix)
        print("[INFO] BPR Model fitting complete. ")
    
    def recommend(self, item_id, n_recommendations=10):
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        item_ids, scores = self.model.similar_items(item_id, N=n_recommendations + 1)
        
        recommendations = []
        for idx, score in zip(item_ids, scores):
            if idx != item_id:
                recommendations.append((idx, score))
        
        return recommendations[:n_recommendations]
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        item_ids, scores = self.model.recommend(
            userid=user_id,
            user_items=self.user_item_matrix[user_id],
            N=n_recommendations
        )
        return list(zip(item_ids, scores))


class SVDRecommender(BaseRecommender):
    """Singular Value Decomposition (SVD) Recommender using scikit-learn."""
    
    def __init__(self, n_components=64, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_item_matrix = None
        self.item_factors = None
        self.user_factors = None
    
    def fit(self, interaction_matrix):
        print(f"[INFO] Fitting SVD Model with {self.n_components} components...")
        self.user_item_matrix = interaction_matrix
        self.user_factors = self.model.fit_transform(self.user_item_matrix)
        self.item_factors = self.model.components_.T
        print("[INFO] SVD Model fitting complete. ✅")
    
    def recommend(self, item_id, n_recommendations=10):
        if self.item_factors is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        target_vector = self.item_factors[item_id].reshape(1, -1)
        similarities = cosine_similarity(target_vector, self.item_factors).flatten()
        
        top_indices = similarities.argsort()[-(n_recommendations + 1):][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx != item_id:
                recommendations.append((idx, similarities[idx]))
        
        return recommendations[:n_recommendations]
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        if self.user_factors is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
        
        user_scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        interacted_items = self.user_item_matrix[user_id].indices
        user_scores[interacted_items] = -np.inf
        
        top_indices = user_scores.argsort()[-n_recommendations:][::-1]
        
        return [(idx, user_scores[idx]) for idx in top_indices]


# ============================================
# MODEL FACTORY
# ============================================
def get_model(model_name: str, params: dict):
    """
    Factory function to create a model instance.
    
    Args:
        model_name: Name of the model (ALS, BPR, SVD, ItemKNN)
        params: Model hyperparameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is unknown
    """
    models = {
        "ALS": ALSRecommender,
        "BPR": BPRRecommender,
        "SVD": SVDRecommender,
        "ItemKNN": ItemBasedKNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    if model_name == "ItemKNN":
        params.pop("random_state", None)
        
    return models[model_name](**params)


def get_model_params(config: dict, model_name: str) -> dict:
    """
    Merge common params with model-specific params from config.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of the model
        
    Returns:
        Dictionary of merged model parameters
    """
    if model_name not in config['models']:
        raise ValueError(f"Model {model_name} not in config")
    
    common = config['models']['common']
    specific = config['models'][model_name]
    
    return {**common, **specific}