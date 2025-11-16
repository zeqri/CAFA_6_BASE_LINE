import numpy as np
from typing import Dict, Set, List


def propagate_batch(pred_batch: np.ndarray, parents_map_local: Dict[str, Set[str]], classes_list: List[str], iterations=3):
    """Propagate predictions up GO hierarchy for a batch."""
    B, Mloc = pred_batch.shape
    idx_map = {i: classes_list[i] for i in range(Mloc)}
    term_to_idx_local = {classes_list[i]: i for i in range(Mloc)}
    for _ in range(iterations):
        changed = False
        for child_idx in range(Mloc):
            child_term = idx_map[child_idx]
            child_scores = pred_batch[:, child_idx]
            for pterm in parents_map_local.get(child_term, []):
                pidx = term_to_idx_local[pterm]
                mask = child_scores > pred_batch[:, pidx]
                if mask.any():
                    pred_batch[mask, pidx] = child_scores[mask]
                    changed = True
        if not changed:
            break
    return pred_batch


def get_top_k_predictions(probs, best_thresh, top_k, mlb):
    """Extract top-K predictions from probability scores."""
    if top_k is None:
        idxs = np.where(probs >= best_thresh)[0]
    else:
        idxs = np.argsort(probs)[-top_k:]
        idxs = [int(x) for x in idxs if probs[x] > 1e-6]
    idxs = sorted(idxs, key=lambda x: probs[x], reverse=True)
    
    results = []
    for idx in idxs:
        score = float(probs[idx])
        if score <= 0.0:
            continue
        go_id = mlb.classes_[idx]
        results.append((go_id, score))
    
    return results
