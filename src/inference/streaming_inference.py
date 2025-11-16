# import gc
# import numpy as np
# import torch
# import esm
# from typing import List
# from .prediction import propagate_batch, get_top_k_predictions


# def initialize_esm_model(config):
#     """Initialize ESM model for inference."""
#     model_dir = str(config["PLM_MODEL_NAME_OR_PATH"])
#     try:
#         esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(model_dir)
#     except:
#         esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#     esm_model.eval()
#     if torch.cuda.is_available():
#         esm_model.to(torch.device("cuda"))
#     esm_batch_converter = esm_alphabet.get_batch_converter()
#     return esm_model, esm_batch_converter


# def streaming_inference(model, test_seqs, config, device, best_thresh, mlb, 
#                        restricted_parents, parents_map, USE_PLM, 
#                        esm_model=None, esm_batch_converter=None, tfidf=None):
#     """Perform streaming inference on test sequences."""
#     from ..data.embeddings import embed_batch_return_np
    
#     out_fpath = config["OUTPUT_SUBMISSION"]
#     open(out_fpath, "w").close()
#     out_f = open(out_fpath, "a")
    
#     test_ids = list(test_seqs.keys())
#     N_test = len(test_ids)
#     print(f"[test] Streaming {N_test} test sequences in batches of {config['EMBED_BATCH_SIZE']} (embed) / predict {config['PREDICT_BATCH_SIZE']}")


import os
import gc
import numpy as np
import torch
import esm
from typing import List
import numpy as np
from typing import Dict, Set, List


from .prediction import propagate_batch, get_top_k_predictions
from data.embeddings import embed_batch_return_np


def initialize_esm_model(config):
    """Initialize ESM model for inference."""
    model_dir = str(config["PLM_MODEL_NAME_OR_PATH"])
    try:
        esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(model_dir)
    except:
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval()
    if torch.cuda.is_available():
        esm_model.to(torch.device("cuda"))
    esm_batch_converter = esm_alphabet.get_batch_converter()
    return esm_model, esm_batch_converter


def streaming_inference(model, test_seqs, config, device, best_thresh, mlb, 
                       restricted_parents, parents_map, USE_PLM, 
                       esm_model=None, esm_batch_converter=None, tfidf=None):
    """Perform streaming inference on test sequences."""
    out_fpath = config["OUTPUT_SUBMISSION"]
    open(out_fpath, "w").close()
    out_f = open(out_fpath, "a")
    
    test_ids = list(test_seqs.keys())
    N_test = len(test_ids)
    print(f"[test] Streaming {N_test} test sequences in batches of {config['EMBED_BATCH_SIZE']} (embed) / predict {config['PREDICT_BATCH_SIZE']}")
    
    embed_batch = []
    embed_ids = []
    
    for i in range(0, N_test, config["EMBED_BATCH_SIZE"]):
        batch_ids = test_ids[i:i+config["EMBED_BATCH_SIZE"]]
        seqs_batch = [test_seqs[pid] for pid in batch_ids]
        emb_mini = embed_batch_return_np(seqs_batch, USE_PLM, esm_model, esm_batch_converter, tfidf)
        embed_batch.append(emb_mini)
        embed_ids.extend(batch_ids)
        buffered_examples = sum(arr.shape[0] for arr in embed_batch)
        if buffered_examples >= config["PREDICT_BATCH_SIZE"] or (i+config["EMBED_BATCH_SIZE"] >= N_test):
            X_buffer = np.vstack(embed_batch).astype(np.float32)
            
            model.eval()
            with torch.no_grad():
                X_buffer_tensor = torch.FloatTensor(X_buffer).to(device)
                y_buffer_prob = model(X_buffer_tensor).cpu().numpy()
            
            if config["PROPAGATE_PREDICTIONS"] and parents_map:
                y_buffer_prob = propagate_batch(y_buffer_prob, restricted_parents, list(mlb.classes_), iterations=3)
            
            for ridx, pid in enumerate(embed_ids):
                probs = y_buffer_prob[ridx]
                predictions = get_top_k_predictions(probs, best_thresh, config["TOP_K_PER_PROTEIN"], mlb)
                for go_id, score in predictions:
                    out_f.write(f"{pid}\t{go_id}\t{score:.3f}\n")
            out_f.flush()
            
            del X_buffer, X_buffer_tensor, y_buffer_prob, embed_batch
            embed_batch = []
            embed_ids = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if (i // config["EMBED_BATCH_SIZE"]) % 50 == 0:
            print(f"[stream] processed {i} / {N_test}")
    
    out_f.close()
    print(f"[done] Submission written to {config['OUTPUT_SUBMISSION']}") 


def streaming_inference_embeddings(model, test_seqs, config, device, best_thresh, mlb, 
                                   restricted_parents=None, parents_map=None, save_path =None):
    """Perform streaming inference on test embeddings (precomputed)."""
    
    out_fpath = os.path.join(save_path,config["OUTPUT_SUBMISSION"])
    open(out_fpath, "w").close()
    out_f = open(out_fpath, "a")
    
    test_ids = list(test_seqs.keys())
    N_test = len(test_ids)
    print(f"[test] Streaming {N_test} test sequences in batches of {config['PREDICT_BATCH_SIZE']}")
    
    embed_batch = []
    embed_ids = []
    
    for i in range(0, N_test, config["PREDICT_BATCH_SIZE"]):
        batch_ids = test_ids[i:i+config["PREDICT_BATCH_SIZE"]]
        # Use the precomputed embeddings directly
        X_buffer = np.vstack([test_seqs[pid] for pid in batch_ids]).astype(np.float32)
        
        model.eval()
        with torch.no_grad():
            X_buffer_tensor = torch.FloatTensor(X_buffer).to(device)
            y_buffer_prob = model(X_buffer_tensor).cpu().numpy()
        
        if config.get("PROPAGATE_PREDICTIONS", False) and parents_map:
            y_buffer_prob = propagate_batch(y_buffer_prob, restricted_parents, list(mlb.classes_), iterations=3)
        
        for ridx, pid in enumerate(batch_ids):
            probs = y_buffer_prob[ridx]
            predictions = get_top_k_predictions(probs, best_thresh, config["TOP_K_PER_PROTEIN"], mlb)
            for go_id, score in predictions:
                out_f.write(f"{pid}\t{go_id}\t{score:.3f}\n")
        out_f.flush()
        
        del X_buffer, X_buffer_tensor, y_buffer_prob
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (i // config["PREDICT_BATCH_SIZE"]) % 50 == 0:
            print(f"[stream] processed {i} / {N_test}")
    
    out_f.close()
    print(f"[done] Submission written to {config['OUTPUT_SUBMISSION']}")
