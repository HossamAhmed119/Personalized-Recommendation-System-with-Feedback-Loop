import numpy as np
from tqdm import tqdm


def evaluate_model_at_k(model, val_matrix, users_to_evaluate, k=10):
    """
    Evaluates the recommender model using Precision, Adjusted Precision,
    Recall, Hit Rate, MRR, and NDCG.
    
    Works with any model implementing the BaseRecommender interface
    (recommend_for_user method). Filtering of already-seen items during
    training is handled internally by the model itself.
    
    Args:
        model: Trained recommender model implementing BaseRecommender.
        val_matrix: Sparse matrix containing validation/test interactions.
        users_to_evaluate: List or array of user IDs to test.
        k: Number of top recommendations to retrieve.
        
    Returns:
        Dictionary containing calculated metrics.
    """
    hits_count = 0
    raw_precisions = []
    adj_precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    for user_id in tqdm(users_to_evaluate, desc="Evaluating Users", leave=False):
        actual_items = set(val_matrix[user_id].indices)
        if len(actual_items) == 0:
            continue

        try:
            # Unified interface — works for ALS, KNN, SVD, BPR
            recs = model.recommend_for_user(user_id, n_recommendations=k)
            recommended_items = [item_id for item_id, score in recs]
        except Exception as e:
            continue

        hits_in_k = [1 if item in actual_items else 0 for item in recommended_items]
        num_hits = sum(hits_in_k)

        # Hit Rate
        if num_hits > 0:
            hits_count += 1

        # Standard Precision@K
        raw_precision = num_hits / k
        raw_precisions.append(raw_precision)

        # Adjusted Precision@K
        possible_hits = min(k, len(actual_items))
        adj_precision = num_hits / possible_hits if possible_hits > 0 else 0
        adj_precisions.append(adj_precision)

        # Recall@K
        recall = num_hits / len(actual_items)
        recalls.append(recall)

        # MRR@K
        mrr = 0.0
        for rank, is_hit in enumerate(hits_in_k, start=1):
            if is_hit:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        # NDCG@K
        dcg = sum([1.0 / np.log2(rank + 1) for rank, is_hit in enumerate(hits_in_k, start=1) if is_hit])
        idcg = sum([1.0 / np.log2(rank + 1) for rank in range(1, possible_hits + 1)])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f'HitRate_{k}': float(np.round(hits_count / len(users_to_evaluate), 4)),
        f'Precision_{k}': float(np.round(np.mean(raw_precisions), 4)),
        f'Adj_Precision_{k}': float(np.round(np.mean(adj_precisions), 4)),
        f'Recall_{k}': float(np.round(np.mean(recalls), 4)),
        f'MRR_{k}': float(np.round(np.mean(mrrs), 4)),
        f'NDCG_{k}': float(np.round(np.mean(ndcgs), 4))
    }