import numpy as np
import torch


def weighted_precision_recall_f1(y_true, y_pred, ia_weights, mlb):
    """Calculate weighted precision, recall, and F1 score."""
    eps = 1e-12
    cls = mlb.classes_
    prec = np.zeros(len(cls), dtype=float)
    rec = np.zeros(len(cls), dtype=float)
    f1 = np.zeros(len(cls), dtype=float)
    
    for i in range(len(cls)):
        tp = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        fn = ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()
        prec[i] = tp / (tp + fp + eps)
        rec[i] = tp / (tp + fn + eps)
        f1[i] = 2 * prec[i] * rec[i] / (prec[i] + rec[i] + eps)
    
    weights = np.array([ia_weights.get(c, 1.0) for c in cls], dtype=float) if 'ia_weights' in globals() else np.ones(len(cls))
    weighted_f1 = (f1 * weights).sum() / (weights.sum() + eps)
    weighted_prec = (prec * weights).sum() / (weights.sum() + eps)
    weighted_rec = (rec * weights).sum() / (weights.sum() + eps)
    return weighted_prec, weighted_rec, weighted_f1


def evaluate_model(model, X_val, y_val, device):
    """Evaluate model on validation set."""
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_prob = model(X_val_tensor).cpu().numpy()
    return y_val_prob


def find_best_threshold(y_val, y_val_prob, ia_weights, mlb, config):
    """Find best threshold for predictions using grid search."""
    best_thresh = 0.5
    best_score = -1.0
    if config["GLOBAL_THRESHOLD_SEARCH"]:
        for t in config["THRESHOLD_GRID"]:
            y_pred_bin = (y_val_prob >= t).astype(int)
            _, _, f1 = weighted_precision_recall_f1(y_val, y_pred_bin, ia_weights, mlb)
            if f1 > best_score:
                best_score = f1
                best_thresh = t
    
    print(f"[eval] best_thresh {best_thresh} best IA-weighted F1 {best_score}")
    return best_thresh, best_score
