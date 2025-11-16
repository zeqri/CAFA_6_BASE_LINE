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



def prepare_label_matrix_and_embeddings(train_proteins, train_terms, train_seqs, chosen_terms):
    """
    Prepare embeddings and binarized label matrix for training.

    Args:
        train_proteins (list): List of protein IDs.
        train_terms (dict): Dictionary mapping protein IDs to label lists.
        train_seqs (dict): Dictionary mapping protein IDs to embeddings (numpy arrays).
        chosen_terms (list): List of all possible terms to binarize.

    Returns:
        Xembeds (np.ndarray): Array of embeddings corresponding to proteins.
        Y (np.ndarray): Binarized label matrix.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer object.
    """
    # Collect embeddings in the same order as train_proteins
    Xembeds = np.array([train_seqs[p] for p in train_proteins], dtype=np.float32)
    
    # Collect labels
    y_labels = [train_terms[p] for p in train_proteins]
    
    # Binarize labels
    mlb = MultiLabelBinarizer(classes=sorted(chosen_terms))
    Y = mlb.fit_transform(y_labels).astype(np.float32)
    
    print("[prep] Embeddings shape:", Xembeds.shape)
    print("[prep] Label matrix shape:", Y.shape)
    
    return Xembeds, Y, mlb

