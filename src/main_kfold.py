import os
import random
import argparse
import pandas as pd
import numpy as np
import torch

# Import configuration
from config import CONFIG

# Import data functions
from data import (
    read_train_terms, parse_obo, read_IA_safe,
    select_top_k_labels, filter_terms_by_chosen, 
    create_term_mappings,  prepare_label_matrix_and_embeddings, get_tax_dict
)

# Import model functions
from models import (
    create_model, train_model_kfold, create_data_loaders,
    evaluate_model, find_best_threshold
) 

# Import inference functions
from inference import streaming_inference_embeddings

# Import utility functions
from utils import propagate_labels_up_hierarchy, build_restricted_parents_map


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensemble_submissions(submission_files, weights, output_path):
    """
    Ensemble multiple submission files with given weights.
    
    Args:
        submission_files: List of paths to submission TSV files
        weights: List of weights for each submission (should sum to 1)
        output_path: Path to save the ensembled submission
    """
    print(f"\n[ensemble] Ensembling {len(submission_files)} submissions...")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Load first submission
    merged = pd.read_csv(
        submission_files[0],
        sep='\t', 
        header=None, 
        names=['Id', 'GO term', 'Confidence']
    )
    merged['Confidence'] = merged['Confidence'] * weights[0]
    
    # Merge remaining submissions
    for i, sub_file in enumerate(submission_files[1:], 1):
        submission = pd.read_csv(
            sub_file,
            sep='\t', 
            header=None, 
            names=['Id', 'GO term', 'Confidence']
        )
        
        merged = merged.merge(
            submission,
            on=['Id', 'GO term'],
            how='outer',
            suffixes=('', f'_{i}')
        )
        
        # Fill NaN with 0 and add weighted contribution
        confidence_col = f'Confidence_{i}' if i > 1 else 'Confidence'
        merged[confidence_col] = merged[confidence_col].fillna(0)
        merged['Confidence'] = merged['Confidence'].fillna(0) + weights[i] * merged[confidence_col]
    
    # Keep only necessary columns
    final_submission = merged[['Id', 'GO term', 'Confidence']]
    
    # Save ensembled submission
    final_submission.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"[ensemble] Saved ensembled submission to {output_path}")
    print(f"[ensemble] Total predictions: {len(final_submission)}")


def main(args):
    """Main execution pipeline with K-fold cross-validation."""
    print("Files are listed!!!")
    print("CUDA available:", torch.cuda.is_available())
    print("config...")
    
    # Set random seeds
    set_random_seeds(CONFIG["RANDOM_SEED"])
    print("import done!!!")

    BASE_PATH = CONFIG["BASE_PATH"]
    SAVE_PATH = f"{BASE_PATH}/models/{args.model_name}"
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # ------------------------------------------------------------
    # Load the data
    # ------------------------------------------------------------
    train_terms_path = os.path.join(CONFIG["EMBED_DIR"], "train_ids.npy")
    train_embeds_path = os.path.join(CONFIG["EMBED_DIR"], "train_embeds.npy")
    train_terms = np.load(train_terms_path, allow_pickle=True)
    train_terms = np.array([term.split('|')[1] for term in train_terms])
    train_embeds = np.load(train_embeds_path)
    train_seqs = {term: embed for term, embed in zip(train_terms, train_embeds)}
    
    train_terms = read_train_terms(CONFIG["TRAIN_TERMS"])
    parents_map, children_map = parse_obo(CONFIG["GO_OBO"])
    train_proteins = [p for p in train_terms.keys() if p in train_seqs]
    print(f"[io] {len(train_proteins)} train proteins with sequences available") 

    test_terms_path = os.path.join(CONFIG["EMBED_DIR"], "test_ids.npy")
    test_embeds_path = os.path.join(CONFIG["EMBED_DIR"], "test_embeds.npy")
    test_terms = np.load(test_terms_path, allow_pickle=True)
    test_embeds = np.load(test_embeds_path)
    test_seqs = {term: embed for term, embed in zip(test_terms, test_embeds)} 

    if CONFIG["USE_TAX"]:
        feather_path_train = os.path.join(CONFIG["HELPERS_PATH"], 'fasta/train_seq.feather')
        feather_df_train = pd.read_feather(feather_path_train)
        tax_dict_train = get_tax_dict(feather_df_train) 

        train_seqs = {
            key: np.concatenate([train_seqs[key], tax_dict_train[key]]).astype(np.float32)
            for key in train_seqs
        }
        
        # Test set
        feather_path_test = os.path.join(CONFIG["HELPERS_PATH"], 'fasta/test_seq.feather')
        feather_df_test = pd.read_feather(feather_path_test)
        tax_dict_test = get_tax_dict(feather_df_test)
        test_seqs = {
            key: np.concatenate([test_seqs[key], tax_dict_test[key]]).astype(np.float32)
            for key in test_seqs
        }

    # Propagate train labels
    if CONFIG["PROPAGATE_TRAIN_LABELS"] and parents_map:
        train_terms = propagate_labels_up_hierarchy(train_proteins, train_terms, parents_map)
    
    # Choose top-k labels
    chosen_terms = select_top_k_labels(train_proteins, train_terms, CONFIG["TOP_K_LABELS"])
    train_terms = filter_terms_by_chosen(train_proteins, train_terms, chosen_terms)
    
    # Prepare label matrix
    X_embeds_train, Y, mlb = prepare_label_matrix_and_embeddings(
        train_proteins=train_proteins,
        train_terms=train_terms,
        train_seqs=train_seqs,
        chosen_terms=chosen_terms
    )
    
    # ------------------------------------------------------------
    # K-Fold Cross-Validation Training
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using device: {device}")
    
    # Train models using K-fold cross-validation
    fold_results = train_model_kfold(
        X_embeds_train, 
        Y, 
        CONFIG, 
        device, 
        SAVE_PATH,
        model_creator_fn=create_model
    )
    
    # ------------------------------------------------------------
    # Generate submissions for each fold
    # ------------------------------------------------------------
    ia_weights = read_IA_safe(CONFIG["IA_FILE"])
    term_to_idx, idx_to_term = create_term_mappings(mlb)
    restricted_parents = build_restricted_parents_map(mlb.classes_, parents_map, term_to_idx)
    
    submission_files = []
    
    for fold_num, (model, val_indices) in enumerate(
        zip(fold_results['models'], fold_results['val_indices']), 1
    ):
        print(f"\n[inference] Generating submission for fold {fold_num}")
        
        # Find best threshold on validation set for this fold
        X_val_fold = X_embeds_train[val_indices]
        y_val_fold = Y[val_indices]
        y_val_prob = evaluate_model(model, X_val_fold, y_val_fold, device)
        best_thresh, best_score = find_best_threshold(
            y_val_fold, y_val_prob, ia_weights, mlb, CONFIG
        )
        print(f"[inference] Fold {fold_num} - Best threshold: {best_thresh:.4f}, Score: {best_score:.4f}")
        
        # Generate submission for this fold
        fold_save_path = os.path.join(SAVE_PATH, f"fold_{fold_num}")
        os.makedirs(fold_save_path, exist_ok=True)
        
        streaming_inference_embeddings(
            model=model,
            test_seqs=test_seqs,
            config=CONFIG,
            device=device,
            best_thresh=best_thresh,
            mlb=mlb,
            restricted_parents=restricted_parents,
            parents_map=parents_map,
            save_path=fold_save_path
        )
        
        submission_file = os.path.join(fold_save_path, "submission.tsv")
        submission_files.append(submission_file)
    
    # ------------------------------------------------------------
    # Ensemble submissions
    # ------------------------------------------------------------
    if args.ensemble:
        # Use equal weights or inverse validation loss weights
        if args.use_loss_weights:
            # Weight by inverse validation loss (lower loss = higher weight)
            val_losses = np.array(fold_results['val_losses'])
            weights = 1.0 / val_losses
        else:
            # Equal weights
            weights = np.ones(len(submission_files))
        
        ensemble_output = os.path.join(SAVE_PATH, "submission_ensemble.tsv")
        ensemble_submissions(submission_files, weights, ensemble_output)
    
    # ------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------
    np.save(f"{SAVE_PATH}/mlb_classes.npy", np.array(mlb.classes_, dtype=object))
    
    # Save fold information
    fold_info = {
        'val_losses': fold_results['val_losses'],
        'avg_val_loss': np.mean(fold_results['val_losses']),
        'std_val_loss': np.std(fold_results['val_losses'])
    }
    np.save(f"{SAVE_PATH}/fold_info.npy", fold_info)
    
    print("\n[done] All folds trained, submissions generated, and artifacts saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAFA6 model with K-fold CV")
    parser.add_argument(
        "--model_name",
        type=str,
        default="cafa6_kfold_model",
        help="Name to use for saving the trained models and related files"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Generate an ensembled submission from all folds"
    )
    parser.add_argument(
        "--use_loss_weights",
        action="store_true",
        help="Weight ensemble by inverse validation loss (default: equal weights)"
    )
    
    args = parser.parse_args()
    main(args)