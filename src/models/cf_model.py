import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import implicit
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity



class BaseRecommender(ABC):
    """
    Abstract Base Class for all recommendation models.
    """
    def __init__(self, **kwargs):
        self.model = None

    @abstractmethod
    def fit(self, interaction_matrix):
        """Trains the model on the provided interaction matrix."""
        pass

    @abstractmethod
    def recommend(self, target_id, n_recommendations=10):
        """Generates recommendations for a given target ID (user or item)."""
        pass


class ItemBasedKNN(BaseRecommender):
    """
    Item-Based Collaborative Filtering using K-Nearest Neighbors.
    Computes similarities between items based on user interaction patterns.
    """
    def __init__(self, n_neighbors=20, metric='cosine', algorithm='brute'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        # 'brute' is usually required for sparse matrices with cosine similarity
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors, 
            metric=self.metric, 
            algorithm=algorithm,
            n_jobs=-1 
        )
        self.item_matrix = None

    def fit(self, interaction_matrix):
        """
        Trains the KNN model.
        Note: For Item-Based KNN, we need the matrix to be (Items x Users).
        If the input is (Users x Items), it needs to be transposed.
        """
        print(f"[INFO] Fitting Item-Based KNN with metric='{self.metric}'...")
        # Transpose matrix to represent Items as rows and Users as columns
        self.item_matrix = interaction_matrix.T 
        self.model.fit(self.item_matrix)
        print("[INFO] KNN Model fitting complete. ✅")

    def recommend(self, item_id, n_recommendations=10):
        """
        Recommends similar items for a given item_id.
        """
        if self.item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")

        # Extract the vector for the target item
        target_item_vector = self.item_matrix[item_id]
        
        # We query for n_recommendations + 1 because the item itself will be returned as the closest match
        distances, indices = self.model.kneighbors(
            target_item_vector, 
            n_neighbors=n_recommendations + 1
        )

        # Flatten arrays and remove the first item (which is the target_item itself)
        distances = distances.flatten()[1:]
        indices = indices.flatten()[1:]

        recommendations = list(zip(indices, distances))
        return recommendations
    

class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) Recommender using the 'implicit' library.
    Excellent for highly sparse implicit feedback datasets.
    """
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

    def fit(self, interaction_matrix: csr_matrix):
        """
        Trains the ALS model.
        Expects a user-by-item CSR matrix.
        """
        print(f"[INFO] Fitting ALS Model with {self.factors} latent factors...")
        self.user_item_matrix = interaction_matrix
        # implicit handles csr_matrix efficiently
        self.model.fit(self.user_item_matrix)
        print("[INFO] ALS Model fitting complete. ")

    def recommend(self, item_id, n_recommendations=10):
        """
        Returns similar items for a given item_id (Item-to-Item similarity).
        Used here to compare outputs directly with ItemBasedKNN.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")

        # Get N+1 items because the target item itself will be included
        item_ids, scores = self.model.similar_items(item_id, N=n_recommendations + 1)
        
        recommendations = []
        for idx, score in zip(item_ids, scores):
            if idx != item_id:
                recommendations.append((idx, score))
                
        # Return exact number of requested recommendations
        return recommendations[:n_recommendations]

    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Generates personalized item recommendations for a specific user.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
            
        item_ids, scores = self.model.recommend(
            userid=user_id, 
            user_items=self.user_item_matrix[user_id], 
            N=n_recommendations
        )
        return list(zip(item_ids, scores))


class BPRRecommender(BaseRecommender):
    """
    Bayesian Personalized Ranking (BPR) Recommender using 'implicit'.
    Directly optimizes the ranking of items, making it highly effective for implicit feedback.
    """
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

    def fit(self, interaction_matrix: csr_matrix):
        """
        Trains the BPR model.
        """
        print(f"[INFO] Fitting BPR Model with {self.factors} latent factors...")
        self.user_item_matrix = interaction_matrix
        self.model.fit(self.user_item_matrix)
        print("[INFO] BPR Model fitting complete. ✅")

    def recommend(self, item_id, n_recommendations=10):
        """
        Returns similar items for a given item_id.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")

        item_ids, scores = self.model.similar_items(item_id, N=n_recommendations + 1)
        
        recommendations = []
        for idx, score in zip(item_ids, scores):
            if idx != item_id:
                recommendations.append((idx, score))
                
        return recommendations[:n_recommendations]

    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Generates personalized item recommendations for a specific user.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
            
        item_ids, scores = self.model.recommend(
            userid=user_id, 
            user_items=self.user_item_matrix[user_id], 
            N=n_recommendations
        )
        return list(zip(item_ids, scores))
    

class SVDRecommender(BaseRecommender):
    """
    Singular Value Decomposition (SVD) Recommender.
    Uses TruncatedSVD from scikit-learn for dimensionality reduction.
    """
    def __init__(self, n_components=64, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_item_matrix = None
        self.item_factors = None
        self.user_factors = None

    def fit(self, interaction_matrix):
        """
        Trains the SVD model and extracts user and item latent factors.
        """
        print(f"[INFO] Fitting SVD Model with {self.n_components} components...")
        self.user_item_matrix = interaction_matrix
        
        # Fit the model and get User Features (U * Sigma)
        self.user_factors = self.model.fit_transform(self.user_item_matrix)
        
        # Get Item Features (V^T) and transpose so items are rows
        self.item_factors = self.model.components_.T 
        print("[INFO] SVD Model fitting complete. ✅")

    def recommend(self, item_id, n_recommendations=10):
        """
        Returns similar items based on the Cosine Similarity of their latent factors.
        """
        if self.item_factors is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")

        # Extract the target item vector
        target_vector = self.item_factors[item_id].reshape(1, -1)
        
        # Compute cosine similarity with all other items
        similarities = cosine_similarity(target_vector, self.item_factors).flatten()
        
        # Get indices of the top similarities (excluding the item itself)
        # argsort sorts ascending, so we take the last N+1 and reverse it
        top_indices = similarities.argsort()[-(n_recommendations + 1):][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx != item_id:
                recommendations.append((idx, similarities[idx]))
                
        return recommendations[:n_recommendations]

    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Generates personalized recommendations by reconstructing the matrix.
        """
        if self.user_factors is None:
            raise ValueError("Model has not been trained. Call 'fit' first.")
            
        # Reconstruct user scores: dot product of user vector and all item vectors
        user_scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        # Get items the user already interacted with to exclude them
        interacted_items = self.user_item_matrix[user_id].indices
        user_scores[interacted_items] = -np.inf 
        
        top_indices = user_scores.argsort()[-n_recommendations:][::-1]
        
        return [(idx, user_scores[idx]) for idx in top_indices]