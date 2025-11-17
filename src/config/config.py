# TSV FILES

BASE_PATH="/home/fr/fr_fr/fr_ka134/kaggle/CAFA_6_BASE_LINE"

SAMPLE_SUBMISSION_TSV = f"{BASE_PATH}/cafa-6-protein-function-prediction/sample_submission.tsv"
IA_TSV = f"{BASE_PATH}/cafa-6-protein-function-prediction/IA.tsv"
TESTSUPERSET_TAXON_LIST_TSV = f"{BASE_PATH}/cafa-6-protein-function-prediction/Test/testsuperset-taxon-list.tsv"
TRAIN_TERMS_TSV = f"{BASE_PATH}/cafa-6-protein-function-prediction/Train/train_terms.tsv"
TRAIN_TAXONOMY_TSV = f"{BASE_PATH}/cafa-6-protein-function-prediction/Train/train_taxonomy.tsv"

# FASTA FILES
TESTSUPERSET_FASTA = f"{BASE_PATH}/cafa-6-protein-function-prediction/Test/testsuperset.fasta"
TRAIN_SEQUENCES_FASTA = f"{BASE_PATH}/cafa-6-protein-function-prediction/Train/train_sequences.fasta"

# OBO FILE
GO_BASIC_OBO = f"{BASE_PATH}/cafa-6-protein-function-prediction/Train/go-basic.obo"
EMBEDS_PATH=f"{BASE_PATH}/embeds"
HELPERS_PATH=f"{BASE_PATH}/helpers"

# OUTPUT FILE
OUTPUT_TSV = "submission.tsv"

# Configuration dictionary
CONFIG = {
    "BASE_PATH": BASE_PATH,
    "TRAIN_FASTA": TRAIN_SEQUENCES_FASTA,
    "TRAIN_TERMS": TRAIN_TERMS_TSV,
    "TRAIN_TAXONOMY": TRAIN_TAXONOMY_TSV,
    "GO_OBO": GO_BASIC_OBO,
    "IA_FILE": IA_TSV,
    "TEST_FASTA": TESTSUPERSET_FASTA,
    "TEST_TAXON_LIST": TESTSUPERSET_TAXON_LIST_TSV,
    "SAMPLE_SUBMISSION": SAMPLE_SUBMISSION_TSV,
    "OUTPUT_SUBMISSION": OUTPUT_TSV,
    "USE_PLM_MODEL": True,
    # "PLM_MODEL_NAME_OR_PATH": "/kaggle/input/esm-2/keras/esm2_t6_8m/1",
    "PLM_MODEL_NAME_OR_PATH": "/home/fr/fr_fr/fr_ka134/kaggle/esm2_models/esm2_t33_650M_UR50D.pt", 
    "EMBED_DIR": EMBEDS_PATH,
    "HELPERS_PATH": HELPERS_PATH,
    "PLM_BATCH_SIZE": 2,
    "EMBED_BATCH_SIZE": 2,
    "PREDICT_BATCH_SIZE": 64,
    "TOP_K_LABELS": 3000,
    "RANDOM_SEED": 42,
    "BATCH_SIZE": 32,
    "EPOCHS": 200,
    "LEARNING_RATE": 1e-3,
    "HIDDEN_UNITS": 512,
    "DROPOUT": 0.2,
    "TOP_K_PER_PROTEIN": 200,
    "GLOBAL_THRESHOLD_SEARCH": True,
    "THRESHOLD_GRID": [i/100 for i in range(1, 51)],
    "PROPAGATE_TRAIN_LABELS": True,
    "PROPAGATE_PREDICTIONS": True,
    "TRAIN_EMB_MEMMAP": "models/train_embs.memmap",
    "TRAIN_EMB_SHAPE_FILE": "models/train_embs_shape.npy",
    "USE_TAX": False
}
