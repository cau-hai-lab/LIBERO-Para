#!/usr/bin/env python3
"""
Add similarity scores to verified CSV

기존 compute_similarity.py의 로직을 사용해서 verified_csv에 
structural_similarity, keyword_similarity 등을 추가해서 저장.

Usage:
    python add_similarity_to_verified.py --verified_csv /path/to/verified.csv --output_csv /path/to/output.csv

Requirements:
    pip install spacy sentence-transformers pandas numpy tqdm
    python -m spacy download en_core_web_sm
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Dependency Check
# =============================================================================

def check_dependencies():
    missing = []
    try:
        import spacy
    except ImportError:
        missing.append("spacy")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print("Missing dependencies. Please install:")
        print(f"  pip install {' '.join(missing)}")
        if "spacy" not in missing:
            print("  python -m spacy download en_core_web_sm")
        exit(1)


# =============================================================================
# 1. Tree Edit Distance (Zhang-Shasha Algorithm)
# =============================================================================

class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []


def build_dependency_tree(doc) -> TreeNode:
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
    
    if root_token is None:
        return TreeNode(("ROOT", "X"))
    
    def build_subtree(token):
        label = (token.dep_, token.pos_)
        children = [build_subtree(child) for child in token.children]
        children.sort(key=lambda x: x.label)
        return TreeNode(label, children)
    
    return build_subtree(root_token)


def tree_to_list(node: TreeNode) -> list:
    result = []
    for child in node.children:
        result.extend(tree_to_list(child))
    result.append(node)
    return result


def get_leftmost_leaf_indices(nodes: list) -> list:
    n = len(nodes)
    leftmost = [0] * n
    
    for i, node in enumerate(nodes):
        if not node.children:
            leftmost[i] = i
        else:
            min_idx = i
            for child in node.children:
                child_idx = nodes.index(child)
                if leftmost[child_idx] < min_idx:
                    min_idx = leftmost[child_idx]
            leftmost[i] = min_idx
    
    return leftmost


def zhang_shasha_distance(tree1: TreeNode, tree2: TreeNode) -> int:
    nodes1 = tree_to_list(tree1)
    nodes2 = tree_to_list(tree2)
    
    n1, n2 = len(nodes1), len(nodes2)
    
    if n1 == 0 and n2 == 0:
        return 0
    if n1 == 0:
        return n2
    if n2 == 0:
        return n1
    
    l1 = get_leftmost_leaf_indices(nodes1)
    l2 = get_leftmost_leaf_indices(nodes2)
    
    def get_keyroots(nodes, leftmost):
        keyroots = []
        seen_leftmost = set()
        for i in range(len(nodes) - 1, -1, -1):
            if leftmost[i] not in seen_leftmost:
                keyroots.append(i)
                seen_leftmost.add(leftmost[i])
        return sorted(keyroots)
    
    kr1 = get_keyroots(nodes1, l1)
    kr2 = get_keyroots(nodes2, l2)
    
    TD = np.zeros((n1 + 1, n2 + 1))
    
    def compute_forest_distance(i, j):
        FD = np.zeros((n1 + 1, n2 + 1))
        
        li, lj = l1[i], l2[j]
        FD[li][lj] = 0
        
        for x in range(li, i + 1):
            FD[x + 1][lj] = FD[x][lj] + 1
        
        for y in range(lj, j + 1):
            FD[li][y + 1] = FD[li][y] + 1
        
        for x in range(li, i + 1):
            for y in range(lj, j + 1):
                cost = 0 if nodes1[x].label == nodes2[y].label else 1
                
                if l1[x] == li and l2[y] == lj:
                    FD[x + 1][y + 1] = min(
                        FD[x][y + 1] + 1,
                        FD[x + 1][y] + 1,
                        FD[x][y] + cost
                    )
                    TD[x + 1][y + 1] = FD[x + 1][y + 1]
                else:
                    FD[x + 1][y + 1] = min(
                        FD[x][y + 1] + 1,
                        FD[x + 1][y] + 1,
                        FD[l1[x]][l2[y]] + TD[x + 1][y + 1]
                    )
    
    for i in kr1:
        for j in kr2:
            compute_forest_distance(i, j)
    
    return int(TD[n1][n2])


class StructuralSimilarity:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def compute(self, text1: str, text2: str) -> dict:
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        
        tree1 = build_dependency_tree(doc1)
        tree2 = build_dependency_tree(doc2)
        
        size1 = len(tree_to_list(tree1))
        size2 = len(tree_to_list(tree2))
        
        ted = zhang_shasha_distance(tree1, tree2)
        
        # 변경: max(size1, size2) → size1 + size2
        total_size = size1 + size2
        similarity = 1.0 - (ted / total_size) if total_size > 0 else 1.0
        
        return {
            "tree_edit_distance": ted,
            "structural_similarity": similarity,
            "tree_size_1": size1,
            "tree_size_2": size2,
        }


# =============================================================================
# 2. Keyword Similarity (Max Matching)
# =============================================================================

class KeywordSimilarity:
    def __init__(self, nlp, encoder):
        self.nlp = nlp
        self.encoder = encoder
        self.embedding_cache = {}
    
    def extract_content_words(self, text: str) -> list:
        """
        Extract content words (NOUN, VERB, ADJ, ADV)
        - put, open 같은 task-critical verb 포함
        - is_stop 체크 안 함 (로봇 task에서는 이런 verb가 중요)
        """
        doc = self.nlp(text.lower())
        content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        
        words = []
        for token in doc:
            if token.pos_ in content_pos:
                words.append(token.lemma_)
        
        return words
    
    def get_embeddings_batch(self, words: list) -> np.ndarray:
        if not words:
            return None
        
        uncached = [w for w in words if w not in self.embedding_cache]
        if uncached:
            embeddings = self.encoder.encode(uncached, convert_to_numpy=True)
            for w, emb in zip(uncached, embeddings):
                self.embedding_cache[w] = emb
        
        return np.array([self.embedding_cache[w] for w in words])
    
    def cosine_sim_matrix(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        return np.dot(emb1_norm, emb2_norm.T)
    
    def max_matching_similarity(self, orig_words: list, para_words: list) -> tuple:
        if not orig_words or not para_words:
            return 0.0, [], []
        
        orig_embs = self.get_embeddings_batch(orig_words)
        para_embs = self.get_embeddings_batch(para_words)
        
        sim_matrix = self.cosine_sim_matrix(orig_embs, para_embs)
        
        max_sims = sim_matrix.max(axis=1)
        best_match_idx = sim_matrix.argmax(axis=1)
        best_matches = [(orig_words[i], para_words[best_match_idx[i]], max_sims[i]) 
                        for i in range(len(orig_words))]
        
        return float(max_sims.mean()), max_sims.tolist(), best_matches
    
    def jaccard_sim(self, words1: list, words2: list) -> float:
        set1 = set(words1)
        set2 = set(words2)
        if len(set1 | set2) == 0:
            return 1.0
        return len(set1 & set2) / len(set1 | set2)
    
    def compute(self, text1: str, text2: str) -> dict:
        words1 = self.extract_content_words(text1)
        words2 = self.extract_content_words(text2)
        
        jaccard = self.jaccard_sim(words1, words2)
        keyword_sim, individual_sims, best_matches = self.max_matching_similarity(words1, words2)
        
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        
        nouns1 = [t.lemma_ for t in doc1 if t.pos_ in ["NOUN", "PROPN"]]
        nouns2 = [t.lemma_ for t in doc2 if t.pos_ in ["NOUN", "PROPN"]]
        
        verbs1 = [t.lemma_ for t in doc1 if t.pos_ == "VERB"]
        verbs2 = [t.lemma_ for t in doc2 if t.pos_ == "VERB"]
        
        noun_sim, _, _ = self.max_matching_similarity(nouns1, nouns2)
        verb_sim, _, _ = self.max_matching_similarity(verbs1, verbs2)
        
        return {
            "content_words_1": words1,
            "content_words_2": words2,
            "jaccard_similarity": jaccard,
            "keyword_similarity": keyword_sim,
            "noun_similarity": noun_sim,
            "verb_similarity": verb_sim,
        }


# =============================================================================
# 3. Main Processing
# =============================================================================

def add_similarity_to_csv(verified_csv: str, output_csv: str):
    """verified CSV에 similarity 점수 추가"""
    from tqdm import tqdm
    import spacy
    from sentence_transformers import SentenceTransformer
    
    # Load data
    print(f"Loading CSV: {verified_csv}")
    df = pd.read_csv(verified_csv)
    print(f"Loaded {len(df)} rows")
    
    # Initialize
    print("\nInitializing models...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    print("Loading Sentence-BERT model: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    struct_sim = StructuralSimilarity(nlp)
    keyword_sim = KeywordSimilarity(nlp, encoder)
    
    # Compute similarities
    print("\nComputing similarities...")
    
    structural_similarities = []
    tree_edit_distances = []
    keyword_similarities = []
    jaccard_similarities = []
    noun_similarities = []
    verb_similarities = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original = row['original_instruction']
        paraphrase = row['new_instruction']
        
        # Structural similarity
        struct_result = struct_sim.compute(original, paraphrase)
        structural_similarities.append(struct_result['structural_similarity'])
        tree_edit_distances.append(struct_result['tree_edit_distance'])
        
        # Keyword similarity
        kw_result = keyword_sim.compute(original, paraphrase)
        keyword_similarities.append(kw_result['keyword_similarity'])
        jaccard_similarities.append(kw_result['jaccard_similarity'])
        noun_similarities.append(kw_result['noun_similarity'])
        verb_similarities.append(kw_result['verb_similarity'])
    
    # Add columns to dataframe
    df['structural_similarity'] = structural_similarities
    df['tree_edit_distance'] = tree_edit_distances
    df['keyword_similarity'] = keyword_similarities
    df['jaccard_similarity'] = jaccard_similarities
    df['noun_similarity'] = noun_similarities
    df['verb_similarity'] = verb_similarities
    
    # Create output directory if needed
    output_dir = Path(output_csv).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")
    
    # Print summary
    print("\n" + "="*70)
    print("SIMILARITY SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Structural Similarity: {df['structural_similarity'].mean():.3f} ± {df['structural_similarity'].std():.3f}")
    print(f"Tree Edit Distance:    {df['tree_edit_distance'].mean():.1f} ± {df['tree_edit_distance'].std():.1f}")
    print(f"Keyword Similarity:    {df['keyword_similarity'].mean():.3f} ± {df['keyword_similarity'].std():.3f}")
    print(f"Jaccard Similarity:    {df['jaccard_similarity'].mean():.3f} ± {df['jaccard_similarity'].std():.3f}")
    print(f"Noun Similarity:       {df['noun_similarity'].mean():.3f} ± {df['noun_similarity'].std():.3f}")
    print(f"Verb Similarity:       {df['verb_similarity'].mean():.3f} ± {df['verb_similarity'].std():.3f}")
    
    # By category if available
    if 'high' in df.columns:
        print("\n" + "-"*70)
        print("By HIGH Category:")
        for cat in sorted(df['high'].unique()):
            subset = df[df['high'] == cat]
            print(f"  {cat}: Struct={subset['structural_similarity'].mean():.3f}, Kw={subset['keyword_similarity'].mean():.3f} (n={len(subset)})")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Add similarity scores to verified CSV')
    parser.add_argument('--verified_csv', type=str, required=True, help='Path to verified CSV')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV path (default: adds _with_similarity suffix)')
    args = parser.parse_args()
    
    check_dependencies()
    
    # Default output path
    if args.output_csv is None:
        input_path = Path(args.verified_csv)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = str(input_path.parent / f"{input_path.stem}_with_similarity_{timestamp}.csv")
    else:
        # If output_csv is a directory, generate filename
        output_path = Path(args.output_csv)
        if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
            # It's a directory path
            output_path.mkdir(parents=True, exist_ok=True)
            input_path = Path(args.verified_csv)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_csv = str(output_path / f"{input_path.stem}_with_similarity_{timestamp}.csv")
    
    add_similarity_to_csv(args.verified_csv, args.output_csv)
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()