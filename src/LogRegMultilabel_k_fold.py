import os
import random
import numpy as np
import pandas as pd
import torch
import tqdm
import cupy as cp

from typing import Dict, Set, List
from sklearn.model_selection import StratifiedKFold

# --- Project Imports ---
from config import CONFIG

from data import (
    read_fasta, read_train_terms, parse_obo, read_IA_safe,
    select_top_k_labels, filter_terms_by_chosen,
    prepare_label_matrix_and_embeddings, create_term_mappings,
    get_tax_dict
)

from models import (
    split_train_validation, 
    find_best_threshold
)

from inference import streaming_inference_logreg

from utils import (
    propagate_labels_up_hierarchy, build_restricted_parents_map
)


# ---------------------------------------------------------------------
# ------------------------ Reproducibility Setup -----------------------
# ---------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        cp.random.seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------
# ----------------------- Logistic Regression Model --------------------
# ---------------------------------------------------------------------

class LogRegMultilabel:
    def __init__(self, alpha=0.001, max_iter=20, lr=1, tol=1e-5, output_batch=100, intercept_scaling=1):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.output_batch = output_batch
        self.intercept_scaling = intercept_scaling
        self.weights = None

    def fit(self, X, y):
        nrows, ncols = X.shape

        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):

            # Prepare features
            X_ = np.ones((nrows, ncols + 1), dtype=np.float32)
            X[:, 0] *= self.intercept_scaling
            X_[:, 1:] = X
            X = cp.asarray(X_)
            ncols += 1

            # Prepare labels
            if len(y.shape) == 1:
                y = y[:, np.newaxis]

            weights = []

            for i in tqdm.tqdm(range(0, y.shape[1], self.output_batch)):
                y_ = cp.asarray(y[:, i:i + self.output_batch], dtype=np.float32)
                _, nout = y_.shape
                mask = ~cp.isnan(y_)

                w = cp.zeros((ncols, nout), dtype=np.float32)

                for _ in range(self.max_iter):
                    pred = cp.dot(X, w)
                    p = 1 / (1 + cp.exp(-pred))

                    grad = cp.empty((ncols, nout), dtype=cp.float32)
                    cp.dot(X.T, cp.where(mask, p - y_, 0), out=grad)

                    grad = grad / mask.sum(axis=0).astype(np.float32) + self.alpha * w

                    delta = cp.zeros((ncols, nout), dtype=np.float32)

                    for k in range(nout):
                        idx = cp.nonzero(mask[:, k])[0]
                        p_sl = p[idx][:, [k]]
                        X_sl = X[idx]

                        hess = cp.dot((X_sl * p_sl * (1 - p_sl)).T, X_sl)
                        hess = hess / idx.shape[0] + cp.diag(cp.ones(ncols, dtype=np.float32) * self.alpha)

                        delta[:, k] = cp.dot(cp.linalg.inv(hess), grad[:, k])

                    if (delta ** 2).sum() < self.tol:
                        break

                    w -= self.lr * delta

                weights.append(w.get())

            self.weights = np.concatenate(weights, axis=1)

        mempool.free_all_blocks()
        return self

    def predict(self, X, batch_size=10000):
        mempool = cp.cuda.MemoryPool()
        res = np.empty((X.shape[0], self.weights.shape[1]), dtype=np.float32)

        with cp.cuda.using_allocator(allocator=mempool.malloc):
            interc, w = cp.asarray(self.weights[0]) * self.intercept_scaling, cp.asarray(self.weights[1:])

            for i in tqdm.tqdm(range(0, X.shape[0], batch_size)):
                batch = cp.asarray(X[i:i + batch_size], dtype=np.float32)
                pred = cp.dot(batch, w) + interc
                res[i:i + batch_size] = 1 / (1 + cp.exp(-pred)).get()

        mempool.free_all_blocks()
        return res


# ---------------------------------------------------------------------
# --------------------------- MAIN WORKFLOW ----------------------------
# ---------------------------------------------------------------------

def main( num_folds: int = 5):

    seed = CONFIG.get("SEED", 42)
    print(f"\n🔒 Using reproducible seed: {seed}")
    set_seed(seed)

    # print("\nLoading FASTA and taxonomy...")
    # fasta_path = os.path.join(CONFIG["HELPERS_PATH"], 'fasta/train_seq.feather')
    # fasta = pd.read_feather(fasta_path)

    print("\nLoading embeddings...")
    train_terms_path = os.path.join(CONFIG["EMBED_DIR"], "t5large_embeddings_output/train_ids.npy")
    train_embeds_path = os.path.join(CONFIG["EMBED_DIR"], "t5large_embeddings_output/train_embeds.npy")

    train_terms = np.load(train_terms_path, allow_pickle=True)
    train_embeds = np.load(train_embeds_path)
    train_seqs = dict(zip(train_terms, train_embeds))

    print("\nLoading ontology and labels...")
    train_terms = read_train_terms(CONFIG["TRAIN_TERMS"])
    parents_map, children_map = parse_obo(CONFIG["GO_OBO"])
    train_proteins = [p for p in train_terms if p in train_seqs]

    if CONFIG["PROPAGATE_TRAIN_LABELS"]:
        train_terms = propagate_labels_up_hierarchy(train_proteins, train_terms, parents_map)

    print("\nSelecting labels...")
    chosen_terms = select_top_k_labels(train_proteins, train_terms, CONFIG["TOP_K_LABELS"])
    train_terms = filter_terms_by_chosen(train_proteins, train_terms, chosen_terms)

    print("\nBuilding dataset...")
    X_embeds_train, Y, mlb = prepare_label_matrix_and_embeddings(
        train_proteins, train_terms, train_seqs, chosen_terms
    )

    print(f"\nStarting {num_folds}-Fold Validation...\n")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    thresholds = []
    scores = []

    human_labels = np.argmax(Y, axis=1)  # Needed because multilabel cannot be stratified, workaround

    for fold, (train_index, val_index) in enumerate(skf.split(X_embeds_train, human_labels)):
        print(f"\n---- Fold {fold+1}/{num_folds} ----")

        X_train, X_val = X_embeds_train[train_index], X_embeds_train[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        model = LogRegMultilabel(alpha=0.00001)
        model.fit(X_train, y_train)

        y_val_prob = model.predict(X_val)
        ia_weights = read_IA_safe(CONFIG["IA_FILE"])
        best_thresh, best_score = find_best_threshold(y_val, y_val_prob, ia_weights, mlb, CONFIG)

        print(f"Fold {fold+1}: Threshold={best_thresh:.4f}, Score={best_score:.4f}")

        thresholds.append(best_thresh)
        scores.append(best_score)

    final_threshold = np.mean(thresholds)

    print(f"\n****** Cross-Validation Complete ******")
    print(f"Average threshold: {final_threshold:.4f}")
    print(f"Average score: {np.mean(scores):.4f}")

    print("\nRunning inference on full test set...")
    test_terms = np.load(os.path.join(CONFIG["EMBED_DIR"], "t5large_embeddings_output/test_ids.npy"), allow_pickle=True)
    test_embeds = np.load(os.path.join(CONFIG["EMBED_DIR"], "t5large_embeddings_output/test_embeds.npy"))
    test_seqs = dict(zip(test_terms, test_embeds))

    term_to_idx, idx_to_term = create_term_mappings(mlb)
    restricted_parents = build_restricted_parents_map(mlb.classes_, parents_map, term_to_idx)

    save_path = os.path.join(CONFIG["BASE_PATH"], f"models/logreg/fold_{fold}")
    os.makedirs(save_path, exist_ok=True)

    # Train final model on ALL data
    print("\nTraining final model on full dataset...")
    final_model = LogRegMultilabel(alpha=0.00001)
    final_model.fit(X_embeds_train, Y)

    streaming_inference_logreg(
        final_model, test_seqs, CONFIG,
        final_threshold, mlb, restricted_parents,
        parents_map, save_path
    )

    print("\n✓ Training + Inference Complete!")


# ---------------------------------------------------------------------
# ----------------------------- ENTRY POINT ---------------------------
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main( num_folds=5)
