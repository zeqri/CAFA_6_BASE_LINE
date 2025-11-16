import os
import gc
import numpy as np
import torch
import esm
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


def embed_with_plm_to_memmap(all_seq_ids: List[str],
                             seqs_dict: Dict[str, str],
                             memmap_path: str,
                             shape_file: str,
                             config: dict):
    """Embed sequences using PLM and save to memmap file."""
    model_dir = str(config["PLM_MODEL_NAME_OR_PATH"])
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_dir)
    except:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    
    test_seq = list(seqs_dict.values())[0]
    test_data = [("test", test_seq)]
    _, test_sequences, test_tokens = batch_converter(test_data)
    test_tokens = test_tokens.to(device)
    with torch.no_grad():
        test_res = model(test_tokens, repr_layers=[model.num_layers], return_contacts=False)
        test_repr = test_res["representations"][model.num_layers][0, 1:len(test_seq)+1, :]
    embed_dim = test_repr.shape[-1]
    del test_tokens, test_res, test_repr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    N = len(all_seq_ids)
    memmap_shape = (N, embed_dim)
    if os.path.exists(memmap_path):
        os.remove(memmap_path)
    emb_memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=memmap_shape)
    np.save(shape_file, np.array(memmap_shape, dtype=int))
    
    batch_size = config["PLM_BATCH_SIZE"]
    for i in range(0, N, batch_size):
        batch_ids = all_seq_ids[i:i+batch_size]
        batch_seqs = [seqs_dict[pid] for pid in batch_ids]
        data = [(pid, seq) for pid, seq in zip(batch_ids, batch_seqs)]
        labels, sequences, tokens = batch_converter(data)
        tokens = tokens.to(device)
        
        with torch.no_grad():
            results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
            repr_keys = sorted(results["representations"].keys())
            last_layer_key = repr_keys[-1]
            repr_tensor = results["representations"][last_layer_key].cpu()
        
        for j, seq in enumerate(sequences):
            seq_len = len(seq)
            seq_repr = repr_tensor[j, 1:seq_len+1, :]
            seq_embed = seq_repr.mean(axis=0).numpy().astype(np.float32)
            emb_memmap[i+j, :] = seq_embed
        
        del tokens, results, repr_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if (i // batch_size) % 10 == 0:
            print(f"[embed] {i}/{N} done")
    
    emb_memmap.flush()
    print(f"[embed] Done. memmap saved to {memmap_path}, shape {memmap_shape}")
    return emb_memmap, memmap_shape


def build_tfidf_embeddings(seq_ids, seqs_dict):
    """Build TF-IDF embeddings for sequences."""
    seqs = [seqs_dict[pid] for pid in seq_ids]
    texts = [" ".join([seq[i:i+3] for i in range(len(seq)-3+1)]) for seq in seqs]
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(texts).astype(np.float32).toarray()
    print(f"[embed] TF-IDF shape: {X_tfidf.shape}")
    return X_tfidf, tfidf


def load_or_create_embeddings(config, X_proteins, train_seqs):
    """Load embeddings from memmap or create new embeddings."""
    USE_PLM = config["USE_PLM_MODEL"] 
    memmap_path = config["TRAIN_EMB_MEMMAP"]
    shape_file = config["TRAIN_EMB_SHAPE_FILE"]
    
    if USE_PLM and os.path.exists(memmap_path) and os.path.exists(shape_file):
        print(f"[embed] Loading existing train embeddings from {memmap_path}")
        shape = tuple(np.load(shape_file))
        X_emb = np.memmap(memmap_path, dtype='float32', mode='r', shape=shape)
        X_train_np = np.array(X_emb, dtype=np.float32)
        tfidf = None
    elif USE_PLM:
        print("[embed] Creating PLM embeddings (this may take some time...)")
        os.makedirs("models", exist_ok=True)
        emb_memmap, shape = embed_with_plm_to_memmap(X_proteins, train_seqs, memmap_path, shape_file, config)
        X_train_np = np.array(emb_memmap, dtype=np.float32)
        tfidf = None
    else:
        print("[embed] Using TF-IDF embeddings (fallback)")
        X_train_np, tfidf = build_tfidf_embeddings(X_proteins, train_seqs)
    
    return X_train_np, tfidf, USE_PLM


def embed_batch_return_np(seq_list: List[str], USE_PLM, esm_model, esm_batch_converter, tfidf):
    """Embed a batch of sequences and return numpy array."""
    if USE_PLM:
        batch_pairs = [(f"seq_{i}", seq) for i, seq in enumerate(seq_list)]
        labels, sequences, tokens = esm_batch_converter(batch_pairs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = tokens.to(device)
        
        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
            repr_keys = sorted(results["representations"].keys())
            last_layer_key = repr_keys[-1]
            repr_tensor = results["representations"][last_layer_key].cpu()
        
        embeddings = []
        for j, seq in enumerate(sequences):
            seq_len = len(seq)
            seq_repr = repr_tensor[j, 1:seq_len+1, :]
            seq_embed = seq_repr.mean(axis=0).numpy().astype(np.float32)
            embeddings.append(seq_embed)
        
        del tokens, results, repr_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return np.array(embeddings, dtype=np.float32)
    else:
        texts = [" ".join([seq[i:i+3] for i in range(len(seq)-3+1)]) for seq in seq_list]
        arr = tfidf.transform(texts).astype(np.float32).toarray()
        return arr
