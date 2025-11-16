import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer


def select_top_k_labels(train_proteins, train_terms, top_k=None):
    """Select top-K most frequent GO terms."""
    all_term_counts = Counter()
    for p in train_proteins:
        all_term_counts.update(train_terms[p])
    all_terms_sorted = [t for t, _ in all_term_counts.most_common()]
    
    if top_k is not None:
        chosen_terms = set(all_terms_sorted[:top_k])
        print(f"[prep] Restricting to top-{top_k} GO terms")
    else:
        chosen_terms = set(all_terms_sorted)
    print(f"[prep] Using {len(chosen_terms)} target GO terms")
    
    return chosen_terms


def filter_terms_by_chosen(train_proteins, train_terms, chosen_terms):
    """Filter training terms to only include chosen terms."""
    for p in train_proteins:
        train_terms[p] = [t for t in train_terms[p] if t in chosen_terms]
    return train_terms


def prepare_label_matrix(train_proteins, train_terms, chosen_terms):
    """Prepare binarized label matrix for training."""
    X_proteins = train_proteins
    y_labels = [train_terms[p] for p in X_proteins]
    
    mlb = MultiLabelBinarizer(classes=sorted(chosen_terms))
    Y = mlb.fit_transform(y_labels).astype(np.float32)
    print("[prep] Label matrix shape:", Y.shape)
    
    return X_proteins, Y, mlb


def create_term_mappings(mlb):
    """Create term-to-index and index-to-term mappings."""
    term_to_idx = {t: i for i, t in enumerate(mlb.classes_)}
    idx_to_term = {i: t for t, i in term_to_idx.items()}
    return term_to_idx, idx_to_term
