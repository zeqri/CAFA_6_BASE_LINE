import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def read_fasta(path: str) -> Dict[str, str]:
    """Read FASTA file and return dictionary of protein IDs to sequences."""
    seqs = {}
    with open(path, "r") as f:
        pid = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(seq_parts)
                header = line[1:].split()[0]
                if "|" in header:
                    parts = header.split("|")
                    pid = parts[1] if len(parts) >= 2 else header
                else:
                    pid = header
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if pid:
            seqs[pid] = "".join(seq_parts)
    print(f"[io] Read {len(seqs)} sequences from {path}")
    return seqs


def read_train_terms(path: str) -> Dict[str, List[str]]:
    """Read training GO term annotations."""
    mapping = defaultdict(list)
    df = pd.read_csv(path, sep="\t", header=None, names=["protein", "go", "ont"], dtype=str)
    for _, r in df.iterrows():
        mapping[r.protein].append(r.go)
    print(f"[io] Read training annotations for {len(mapping)} proteins from {path}")
    return mapping


def parse_obo(go_obo_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Parse OBO file to extract parent-child relationships in GO hierarchy."""
    parents = defaultdict(set)
    children = defaultdict(set)
    if not os.path.exists(go_obo_path):
        return parents, children
    with open(go_obo_path, "r") as f:
        cur_id = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                cur_id = None
            elif line.startswith("id: "):
                cur_id = line.split("id: ")[1].strip()
            elif line.startswith("is_a: "):
                pid = line.split()[1].strip()
                if cur_id:
                    parents[cur_id].add(pid)
                    children[pid].add(cur_id)
            elif line.startswith("relationship: part_of "):
                parts = line.split()
                if len(parts) >= 3:
                    pid = parts[2].strip()
                    if cur_id:
                        parents[cur_id].add(pid)
                        children[pid].add(cur_id)
    print(f"[io] Parsed OBO: {len(parents)} nodes with parents")
    return parents, children


def read_IA_safe(path):
    """Read Information Accretion (IA) weights file."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["go", "ia"], dtype=str)
    d = {}
    for _, r in df.iterrows():
        try:
            d[r.go] = float(r.ia)
        except:
            try:
                d[r.go] = float(r.ia.replace(",", "."))
            except:
                d[r.go] = 0.0
    return d
