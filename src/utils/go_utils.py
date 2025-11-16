from typing import Dict, Set


def get_ancestors(go_id: str, parents: Dict[str, Set[str]]) -> Set[str]:
    """Get all ancestor GO terms for a given GO term."""
    ans = set()
    stack = [go_id]
    while stack:
        cur = stack.pop()
        for p in parents.get(cur, []):
            if p not in ans:
                ans.add(p)
                stack.append(p)
    return ans


def propagate_labels_up_hierarchy(train_proteins, train_terms, parents_map):
    """Propagate training labels up the GO hierarchy."""
    print("[prep] Propagating train labels up GO graph")
    propagated = {}
    for p in train_proteins:
        terms = set(train_terms[p])
        extra = set()
        for t in list(terms):
            extra |= get_ancestors(t, parents_map)
        propagated[p] = sorted(terms | extra)
    return propagated


def build_restricted_parents_map(mlb_classes, parents_map, term_to_idx):
    """Build parents map restricted to chosen terms for faster propagation."""
    restricted_parents = {}
    for t in mlb_classes:
        restricted_parents[t] = set([p for p in parents_map.get(t, set()) if p in term_to_idx])
    return restricted_parents
