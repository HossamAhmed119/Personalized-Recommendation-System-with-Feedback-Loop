import numpy as np
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

def evaluate_model_at_k(model, train_matrix, val_matrix, users_to_evaluate, k=10):
    """
    Evaluates recommender models using professional metrics:
    Standard Precision, Adjusted Precision, Recall, Hit Rate, MRR, and NDCG.
    
    Args:
        model: Trained recommender model (e.g., ALS).
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
        
        # Correct implicit library method for recommendations
        result = model.recommend(
            userid=user_id, 
            user_items=train_matrix[user_id], 
            N=k, 
            filter_already_liked_items=True
        )
        
        # Handle different output formats based on implicit library version
        if isinstance(result, tuple):
            recs = result[0]  # Newer versions return (item_ids, scores)
        else:
            recs = [item for item, score in result] # Older versions return list of tuples
            
        hits_in_k = [1 if item in actual_items else 0 for item in recs]
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
