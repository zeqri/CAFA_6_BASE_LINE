# CAFA-6 Protein Function Prediction

This project implements a protein function prediction system for the CAFA-6 challenge. The code has been modularized into a clean, maintainable structure.

## Project Structure

```
src/
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration settings and file paths
├── data/
│   ├── __init__.py
│   ├── file_io.py                # File reading functions (FASTA, TSV, OBO)
│   ├── preprocessing.py          # Data preparation and label processing
│   └── embeddings.py             # PLM and TF-IDF embedding functions
├── models/
│   ├── __init__.py
│   ├── neural_network.py         # PyTorch model architecture
│   ├── training.py               # Training logic
│   └── evaluation.py             # Evaluation metrics
├── inference/
│   ├── __init__.py
│   ├── prediction.py             # Prediction and propagation logic
│   └── streaming_inference.py   # Streaming test-time inference
├── utils/
│   ├── __init__.py
│   └── go_utils.py               # GO term hierarchy utilities
└── main.py                       # Main execution script
```

## Module Descriptions

### config/
- **config.py**: Contains all file paths and hyperparameters
  - File paths for data files
  - Model configuration (learning rate, batch size, etc.)
  - Feature flags (PLM usage, propagation settings)

### data/
- **file_io.py**: I/O operations
  - `read_fasta()`: Read FASTA sequence files
  - `read_train_terms()`: Read GO term annotations
  - `parse_obo()`: Parse GO ontology file
  - `read_IA_safe()`: Read Information Accretion weights

- **preprocessing.py**: Data preparation
  - `select_top_k_labels()`: Select most frequent GO terms
  - `filter_terms_by_chosen()`: Filter annotations to chosen terms
  - `prepare_label_matrix()`: Create binarized label matrix
  - `create_term_mappings()`: Build term-to-index mappings

- **embeddings.py**: Feature extraction
  - `embed_with_plm_to_memmap()`: Create PLM embeddings using ESM
  - `build_tfidf_embeddings()`: Create TF-IDF embeddings
  - `load_or_create_embeddings()`: Load existing or create new embeddings
  - `embed_batch_return_np()`: Batch embedding function

### models/
- **neural_network.py**: Model architecture
  - `ProteinDataset`: PyTorch dataset class
  - `MultiLabelClassifier`: Neural network model
  - `create_model()`: Model initialization

- **training.py**: Training pipeline
  - `split_train_validation()`: Split data into train/val sets
  - `create_data_loaders()`: Create PyTorch data loaders
  - `train_model()`: Training loop

- **evaluation.py**: Model evaluation
  - `weighted_precision_recall_f1()`: Calculate weighted metrics
  - `evaluate_model()`: Get validation predictions
  - `find_best_threshold()`: Threshold optimization

### inference/
- **prediction.py**: Prediction utilities
  - `propagate_batch()`: Propagate predictions up GO hierarchy
  - `get_top_k_predictions()`: Extract top predictions

- **streaming_inference.py**: Test-time inference
  - `initialize_esm_model()`: Load ESM model for inference
  - `streaming_inference()`: Memory-efficient streaming prediction

### utils/
- **go_utils.py**: GO ontology operations
  - `get_ancestors()`: Get ancestor terms in GO graph
  - `propagate_labels_up_hierarchy()`: Propagate training labels
  - `build_restricted_parents_map()`: Build restricted parent map

## Usage

Run the main pipeline:

```bash
python src/main.py
```

## Key Features

1. **Modular Design**: Each component has a specific responsibility
2. **Clean Separation**: Data, models, and inference logic are separated
3. **Reusability**: Functions can be imported and used independently
4. **Maintainability**: Easy to understand, modify, and extend
5. **Memory Efficient**: Streaming inference for large test sets
6. **Flexible**: Supports both PLM and TF-IDF embeddings

## Configuration

All settings can be modified in `src/config/config.py`:
- Model hyperparameters
- File paths
- Feature flags (PLM usage, label propagation, etc.)
- Batch sizes and thresholds

## Dependencies

- PyTorch
- NumPy
- Pandas
- scikit-learn
- ESM (for protein language models)
