from .file_io import read_fasta, read_train_terms, parse_obo, read_IA_safe
from .preprocessing import (
    select_top_k_labels,
    filter_terms_by_chosen,
    prepare_label_matrix,
    create_term_mappings,
    prepare_label_matrix_and_embeddings
)
from .embeddings import (
    embed_with_plm_to_memmap,
    build_tfidf_embeddings,
    load_or_create_embeddings,
    embed_batch_return_np
)

__all__ = [
    'read_fasta',
    'read_train_terms',
    'parse_obo',
    'read_IA_safe',
    'select_top_k_labels',
    'filter_terms_by_chosen',
    'prepare_label_matrix',
    'prepare_label_matrix_and_embeddings',
    'create_term_mappings',
    'embed_with_plm_to_memmap',
    'build_tfidf_embeddings',
    'load_or_create_embeddings',
    'embed_batch_return_np'
]
