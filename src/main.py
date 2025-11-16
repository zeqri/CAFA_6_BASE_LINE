import os
import random
import numpy as np
import torch

# Import configuration
from config import CONFIG

# Import data functions
from data import (
    read_fasta, read_train_terms, parse_obo, read_IA_safe,
    select_top_k_labels, filter_terms_by_chosen, prepare_label_matrix,
    create_term_mappings, load_or_create_embeddings
)



# Import model functions
from models import (
    create_model, split_train_validation, create_data_loaders,
    train_model, evaluate_model, find_best_threshold
) 


# Import inference functions
from inference import initialize_esm_model, streaming_inference

# Import utility functions
from utils import propagate_labels_up_hierarchy, build_restricted_parents_map


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main execution pipeline."""
    print("Files are listed!!!")
    print("CUDA available:", torch.cuda.is_available())
    print("config...")
    
    # Set random seeds
    set_random_seeds(CONFIG["RANDOM_SEED"])
    print("import done!!!")
    
    # ------------------------------------------------------------
    # Load the data
    # ------------------------------------------------------------
    train_seqs = read_fasta(CONFIG["TRAIN_FASTA"])
    train_terms = read_train_terms(CONFIG["TRAIN_TERMS"])
    parents_map, children_map = parse_obo(CONFIG["GO_OBO"])
    test_seqs = read_fasta(CONFIG["TEST_FASTA"])
    
    train_proteins = [p for p in train_terms.keys() if p in train_seqs]
    print(f"[io] {len(train_proteins)} train proteins with sequences available")
    
    # Propagate train labels
    if CONFIG["PROPAGATE_TRAIN_LABELS"] and parents_map:
        train_terms = propagate_labels_up_hierarchy(train_proteins, train_terms, parents_map)
    
    # Choose top-k labels
    chosen_terms = select_top_k_labels(train_proteins, train_terms, CONFIG["TOP_K_LABELS"])
    train_terms = filter_terms_by_chosen(train_proteins, train_terms, chosen_terms)
    
    # Prepare label matrix
    X_proteins, Y, mlb = prepare_label_matrix(train_proteins, train_terms, chosen_terms)
    
    # ------------------------------------------------------------
    # Create embeddings
    # ------------------------------------------------------------
    X_train_np, tfidf, USE_PLM = load_or_create_embeddings(CONFIG, X_proteins, train_seqs)
    
    # ------------------------------------------------------------
    # Split data and create loaders
    # ------------------------------------------------------------
    X_train, X_val, y_train, y_val = split_train_validation(X_train_np, Y, CONFIG)
    train_loader = create_data_loaders(X_train, y_train, CONFIG)
    
    # ------------------------------------------------------------
    # Create and train model
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using device: {device}")
    
    input_dim = X_train.shape[1]
    num_labels = Y.shape[1]
    model = create_model(input_dim, num_labels, CONFIG, device)
    
    model = train_model(model, train_loader, CONFIG, device)
    
    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    ia_weights = read_IA_safe(CONFIG["IA_FILE"])
    y_val_prob = evaluate_model(model, X_val, y_val, device)
    best_thresh, best_score = find_best_threshold(y_val, y_val_prob, ia_weights, mlb, CONFIG)
    
    # ------------------------------------------------------------
    # Streaming test-time inference
    # ------------------------------------------------------------
    term_to_idx, idx_to_term = create_term_mappings(mlb)
    restricted_parents = build_restricted_parents_map(mlb.classes_, parents_map, term_to_idx)
    
    # Initialize ESM model if using PLM
    esm_model = None
    esm_batch_converter = None
    if USE_PLM:
        esm_model, esm_batch_converter = initialize_esm_model(CONFIG)
    
    # Perform streaming inference
    streaming_inference(
        model, test_seqs, CONFIG, device, best_thresh, mlb,
        restricted_parents, parents_map, USE_PLM,
        esm_model, esm_batch_converter, tfidf
    )
    
    # ------------------------------------------------------------
    # Save model and artifacts
    # ------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cafa6_baseline_model.pt")
    np.save("models/mlb_classes.npy", np.array(mlb.classes_, dtype=object))
    print("[done] saved model and classes; notebook finished.")


if __name__ == "__main__":
    main()
