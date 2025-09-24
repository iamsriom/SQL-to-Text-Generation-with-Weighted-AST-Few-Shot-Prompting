import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sqlparse
from sqlparse.tokens import Keyword
from collections import defaultdict
import heapq
from sqlparse.sql import Function, Identifier

# SQL keywords for feature extraction (same as in feature_importance2.py)
SQL_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN',
    'INNER JOIN', 'OUTER JOIN', 'ON', 'AS', 'AND', 'OR', 'UNION', 'INTERSECT', 'EXCEPT', 'LIMIT',
    'DISTINCT', 'IN', 'NOT', 'IS', 'NULL', 'BETWEEN', 'EXISTS', 'LIKE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
}

def extract_sql_features(query: str) -> List[str]:
    """
    Extract SQL keywords/features from a query using proper multi-word keyword handling.
    Also extract table names (after FROM/JOIN) and aggregation functions (COUNT, SUM, AVG, MIN, MAX).
    """
    parsed = sqlparse.parse(query)
    if not parsed:
        return []
    
    features = []
    
    # Get all tokens
    tokens = list(parsed[0].flatten())
    
    # Extract SQL keywords
    i = 0
    while i < len(tokens):
        if tokens[i].ttype in Keyword:
            # Check for multi-word keywords (2-word combinations)
            if i + 1 < len(tokens) and tokens[i+1].ttype in Keyword:
                two_word = f"{tokens[i].value.upper()} {tokens[i+1].value.upper()}"
                if two_word in SQL_KEYWORDS:
                    features.append(two_word)
                    i += 2  # Skip both tokens
                    continue
            
            # Check for single-word keywords
            if tokens[i].value.upper() in SQL_KEYWORDS:
                features.append(tokens[i].value.upper())
        i += 1
    
    # Use sqlparse's token hierarchy for table names and aggregation functions
    statement = parsed[0]
    agg_functions = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
    for token in statement.tokens:
        # Aggregation functions
        if isinstance(token, Function):
            fname = token.get_name()
            if fname and fname.upper() in agg_functions:
                features.append(f"AGG:{fname.upper()}")
        # Table names after FROM/JOIN
        if token.ttype in Keyword and token.value.upper() in {"FROM", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN"}:
            # Next token should be Identifier or IdentifierList
            idx = statement.token_index(token)
            if idx is not None and idx + 1 < len(statement.tokens):
                next_token = statement.tokens[idx + 1]
                if isinstance(next_token, Identifier):
                    features.append(f"TABLE:{next_token.get_real_name().lower()}")
                elif next_token.is_group:
                    for subtoken in next_token.tokens:
                        if isinstance(subtoken, Identifier):
                            features.append(f"TABLE:{subtoken.get_real_name().lower()}")
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for f in features:
        if f not in seen:
            seen.add(f)
            result.append(f)
    
    return result

def load_feature_weights_v2(weights_file: str) -> Dict[str, float]:
    """
    Load the learned feature weights from v2 JSON file (BoW auxiliary head version).
    """
    with open(weights_file, 'r') as f:
        data = json.load(f)
    
    # Use attention scores from v2 (BoW-supervised) as primary weights
    attention_scores = data.get('attention_scores', {})
    combined_scores = data.get('combined_scores', {})
    
    # Prefer attention scores if available and meaningful
    if attention_scores and any(v > 0 for v in attention_scores.values()):
        print(f"Using BoW-supervised attention weights from {weights_file}")
        return attention_scores
    elif combined_scores:
        print(f"Using combined weights from {weights_file}")
        return combined_scores
    else:
        print(f"Warning: No valid weights found in {weights_file}")
        return {}

def compute_weighted_similarity(query1_features: List[str], query2_features: List[str], 
                              feature_weights: Dict[str, float]) -> float:
    """
    Compute weighted similarity between two queries based on their SQL features.
    Uses the original formula: Σ w(f) · min(c_Q_i(f), c_Q_j(f)) / Σ w(f)
    
    This provides meaningful differentiation between queries without forcing [0,1] range.
    """
    # Convert lists to multisets (count features)
    from collections import Counter
    features1 = Counter(query1_features)
    features2 = Counter(query2_features)
    
    # Get all features from both queries
    all_features = set(features1.keys()) | set(features2.keys())
    
    if not all_features:
        return 0.0
    
    total_similarity = 0.0
    total_weight = 0.0
    
    for feature in all_features:
        count1 = features1.get(feature, 0)
        count2 = features2.get(feature, 0)
        
        # Use minimum count as intersection (as per the formula)
        intersection = min(count1, count2)
        
        # Get weight for this feature (default to 0.0 to avoid bias)
        weight = feature_weights.get(feature, 0.0)
        
        # Only add to totals if weight > 0 (learned features only)
        if weight > 0:
            total_similarity += intersection * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    similarity = total_similarity / total_weight
    return round(similarity, 3)

def load_train_queries(dataset_path: str, dataset_type: str) -> List[Tuple[str, str]]:
    """
    Load queries from train dataset with their translations.
    Returns list of (query, translation) tuples.
    """
    queries = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if dataset_type == 'spider':
        for item in data:
            if 'query' in item and 'question' in item:
                queries.append((item['query'], item['question']))
    elif dataset_type == 'sparc':
        for item in data:
            for turn in item.get('interaction', []):
                if 'query' in turn and 'utterance' in turn:
                    queries.append((turn['query'], turn['utterance']))
    elif dataset_type == 'cosql':
        for item in data:
            for turn in item.get('interaction', []):
                if 'query' in turn and 'utterance' in turn:
                    queries.append((turn['query'], turn['utterance']))
            if 'final' in item and 'query' in item['final'] and 'utterance' in item['final']:
                queries.append((item['final']['query'], item['final']['utterance']))
    
    return queries

def load_dev_queries(dataset_path: str, dataset_type: str) -> List[Tuple[str, str]]:
    """
    Load queries from dev dataset with their translations.
    Returns list of (query, translation) tuples.
    """
    queries = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if dataset_type == 'spider':
        for item in data:
            if 'query' in item and 'question' in item:
                queries.append((item['query'], item['question']))
    elif dataset_type == 'sparc':
        for item in data:
            for turn in item.get('interaction', []):
                if 'query' in turn and 'utterance' in turn:
                    queries.append((turn['query'], turn['utterance']))
    elif dataset_type == 'cosql':
        for item in data:
            for turn in item.get('interaction', []):
                if 'query' in turn and 'utterance' in turn:
                    queries.append((turn['query'], turn['utterance']))
            if 'final' in item and 'query' in item['final'] and 'utterance' in item['final']:
                queries.append((item['final']['query'], item['final']['utterance']))
    
    return queries

def find_top_similar_queries_from_train_v2(target_query: str, train_queries: List[Tuple[str, str]], 
                                         feature_weights: Dict[str, float], top_k: int = 10) -> List[Tuple[float, str, str, int]]:
    """
    Find top-k most similar queries from train dataset to the target dev query.
    FIXED: Proper min-heap implementation that keeps top-K largest similarities.
    
    Returns list of (similarity_score, query, translation, query_index) tuples.
    Ensures unique train queries in results.
    """
    # Use a min-heap to keep top-k largest similarities
    # Store (similarity, index, query, translation) tuples
    # Python's heapq is a min-heap, so we keep the smallest k items
    # When we find a larger similarity, we replace the smallest
    heap = []
    seen_queries = set()  # Track unique queries for deduplication
    
    for idx, (query, translation) in enumerate(train_queries):
        # Skip if we've already seen this exact query (deduplication)
        if query in seen_queries:
            continue
            
        # Extract AST features and compute similarity (as per Algorithm 1)
        similarity = compute_ast_similarity_v2(target_query, query, feature_weights)
        
        # FIXED: Use positive similarity and proper min-heap logic
        if len(heap) < top_k:
            # Heap not full yet, just push
            heapq.heappush(heap, (similarity, idx, query, translation))
        elif similarity > heap[0][0]:
            # Current similarity is larger than smallest in heap
            # Replace the smallest (root of min-heap)
            heapq.heapreplace(heap, (similarity, idx, query, translation))
        
        seen_queries.add(query)
    
    # Extract results from heap, sorted by similarity descending
    top_results = []
    while heap:
        similarity, idx, query, translation = heapq.heappop(heap)
        top_results.append((similarity, query, translation, idx))
    
    # Sort by similarity score descending (largest first)
    top_results.sort(key=lambda x: x[0], reverse=True)
    
    return top_results

def find_top_similar_queries_with_query_attention(target_query: str, 
                                                train_queries: List[Tuple[str, str]], 
                                                idf_weights: Dict[str, float],
                                                attention_analyzer=None,
                                                top_k: int = 10,
                                                alpha: float = 0.5) -> List[Tuple[float, str, str, int]]:
    """
    Find top-k similar queries using query-specific attention mechanism.
    
    This implements the paper's key innovation: attention weights are computed
    specifically for the target query, not globally averaged.
    
    Args:
        target_query: The query to find similar examples for
        train_queries: List of (query, translation) tuples from training set
        idf_weights: Global IDF weights for features
        attention_analyzer: Query-specific attention analyzer
        top_k: Number of top similar queries to return
        alpha: Balance between IDF and attention (0.5 = equal weight)
    
    Returns:
        List of (similarity_score, query, translation, query_index) tuples
    """
    import heapq
    
    # Initialize attention analyzer if not provided
    if attention_analyzer is None:
        try:
            from .query_specific_attention import QuerySpecificAttentionAnalyzer
            attention_analyzer = QuerySpecificAttentionAnalyzer(
                model_path="learned-weights/spider_model.pth",
                vocab_path="learned-weights/spider_feature_importance_v2.json"
            )
        except Exception as e:
            print(f"Warning: Could not initialize query-specific attention: {e}")
            print("Falling back to global weights...")
            return find_top_similar_queries_from_train_v2(
                target_query, train_queries, idf_weights, top_k
            )
    
    heap = []
    seen_queries = set()
    
    for idx, (query, translation) in enumerate(train_queries):
        if query in seen_queries:
            continue
        
        # Compute similarity using query-specific attention
        similarity = compute_ast_similarity_with_query_attention(
            target_query, query, idf_weights, attention_analyzer, alpha
        )
        
        # Use min-heap for top-k selection
        if len(heap) < top_k:
            heapq.heappush(heap, (similarity, idx, query, translation))
        elif similarity > heap[0][0]:
            heapq.heapreplace(heap, (similarity, idx, query, translation))
        
        seen_queries.add(query)
    
    # Extract and sort results
    top_results = []
    while heap:
        similarity, idx, query, translation = heapq.heappop(heap)
        top_results.append((similarity, query, translation, idx))
    
    # Sort by similarity descending
    top_results.sort(key=lambda x: x[0], reverse=True)
    
    return top_results

def process_dev_dataset_v2(dataset_name: str, dev_path: str, train_path: str, weights_file: str, output_dir: str):
    """
    Process a dev dataset to find top similar queries from train dataset for each dev query.
    Uses the v2 weighted similarity system with proper heap logic and AST features.
    Write each dev query's result to a separate file in a subfolder for the dataset.
    Skip files that already exist.
    """
    print(f"Processing {dataset_name} dev dataset with v2 weighted similarity...")
    
    # Load feature weights (v2 format)
    feature_weights = load_feature_weights_v2(weights_file)
    print(f"Loaded {len(feature_weights)} feature weights from v2 system")
    
    # Load train queries (source for similar queries)
    train_queries = load_train_queries(train_path, dataset_name)
    print(f"Loaded {len(train_queries)} train queries")
    
    # Load dev queries (target queries to find similar ones for)
    dev_queries = load_dev_queries(dev_path, dataset_name)
    print(f"Loaded {len(dev_queries)} dev queries")
    
    # Create output directory for this dataset
    dataset_output_path = Path(output_dir) / dataset_name
    dataset_output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each dev query
    processed_count = 0
    skipped_count = 0
    
    for i, (dev_query, dev_translation) in enumerate(dev_queries):
        file_name = f"dev_query_{i+1:05d}.json"
        output_file = dataset_output_path / file_name
        
        # Skip if file already exists
        if output_file.exists():
            print(f"Skipping dev query {i+1}/{len(dev_queries)} - file already exists: {file_name}")
            skipped_count += 1
            continue
            
        print(f"Processing dev query {i+1}/{len(dev_queries)}")
        
        # Find top similar queries from train dataset using v2 system
        similar_queries = find_top_similar_queries_from_train_v2(dev_query, train_queries, feature_weights, top_k=10)
        
        # Store results
        query_result = {
            'dev_query': dev_query,
            'dev_translation': dev_translation,
            'similarity_method': 'v2_weighted_ast_similarity',
            'similar_train_queries': [
                {
                    'similarity_score': float(sim_score),
                    'train_query': sim_query,
                    'train_translation': sim_translation,
                    'train_query_index': sim_idx
                }
                for sim_score, sim_query, sim_translation, sim_idx in similar_queries
            ]
        }
        # Write each dev query's result to a separate file
        with open(output_file, 'w') as f:
            json.dump(query_result, f, indent=2)
        
        processed_count += 1
        
        # Print progress every 50 queries
        if processed_count % 50 == 0:
            print(f"Progress: {processed_count} processed, {skipped_count} skipped")
    
    print(f"Completed: {processed_count} new files written, {skipped_count} files skipped")
    print(f"Total files in {dataset_output_path}: {len(list(dataset_output_path.glob('*.json')))}")
    
    # Print summary statistics
    all_similarities = []
    for i in range(len(dev_queries)):
        file_name = f"dev_query_{i+1:05d}.json"
        output_file = dataset_output_path / file_name
        if not output_file.exists():
            print(f"Warning: {output_file} not found, skipping in summary statistics.")
            continue
        with open(output_file, 'r') as f:
            result = json.load(f)
            for sim in result['similar_train_queries']:
                all_similarities.append(sim['similarity_score'])
    
    if all_similarities:
        print(f"Similarity statistics (v2 system):")
        print(f"  Mean: {np.mean(all_similarities):.3f}")
        print(f"  Median: {np.median(all_similarities):.3f}")
        print(f"  Min: {np.min(all_similarities):.3f}")
        print(f"  Max: {np.max(all_similarities):.3f}")
    
    return None

def compute_ast_similarity_v2(query1: str, query2: str, feature_weights: Dict[str, float]) -> float:
    """
    Compute weighted AST similarity between two SQL queries using the same feature space as training.
    Compares the structure and semantics of the queries using their parse trees.
    
    Uses the same AST feature extraction as feature_importance2.py
    """
    try:
        # Parse both queries
        parsed1 = sqlparse.parse(query1)[0]
        parsed2 = sqlparse.parse(query2)[0]
        
        # Extract AST features from both queries (same as training)
        features1 = extract_ast_features_v2(parsed1)
        features2 = extract_ast_features_v2(parsed2)
        
        # Compute weighted similarity based on AST structure
        similarity = compute_weighted_ast_similarity_v2(features1, features2, feature_weights)
        
        return round(similarity, 3)
    except Exception as e:
        print(f"Error computing AST similarity: {e}")
        return 0.0

def compute_ast_similarity_with_query_attention(query1: str, query2: str, 
                                              idf_weights: Dict[str, float],
                                              attention_analyzer=None,
                                              alpha: float = 0.5) -> float:
    """
    Compute similarity using query-specific attention mechanism as described in the paper.
    
    This implements the key innovation: w(f|Q) = α·IDF(f) + (1-α)·Attn(f|Q)
    where Attn(f|Q) is computed per query, not globally averaged.
    """
    try:
        from .query_specific_attention import QuerySpecificAttentionAnalyzer
        
        # Initialize analyzer if not provided
        if attention_analyzer is None:
            attention_analyzer = QuerySpecificAttentionAnalyzer(
                model_path="learned-weights/spider_model.pth",
                vocab_path="learned-weights/spider_feature_importance_v2.json"
            )
        
        # Use query-specific attention for similarity computation
        similarity = attention_analyzer.compute_weighted_similarity_with_query_attention(
            query1, query2, idf_weights, alpha
        )
        
        return round(similarity, 3)
    except Exception as e:
        print(f"Error computing query-specific attention similarity: {e}")
        # Fallback to global weights
        return compute_ast_similarity_v2(query1, query2, idf_weights)

def extract_ast_features_v2(parsed_query) -> Dict[str, int]:
    """
    Extract AST features from a parsed SQL query with hierarchical relationships.
    IDENTICAL to the feature extraction used in feature_importance2.py training.
    
    Returns a dictionary of feature counts with namespaces: TYPE:, KEYWORD:, FUNCTION:, IDENTIFIER:, DEPTH:
    """
    features = {}
    
    def traverse_ast(node, depth=0, parent_type=None):
        current_type = type(node).__name__
        
        # Extract basic node type features
        features[f"TYPE:{current_type}"] = features.get(f"TYPE:{current_type}", 0) + 1
        
        # Extract depth features (fundamental hierarchical property)
        features[f"DEPTH:{depth}"] = features.get(f"DEPTH:{depth}", 0) + 1
        
        # Extract parent-child relationship features (simple containment)
        if parent_type is not None:
            parent_child_key = f"PARENT_CHILD:{parent_type}->{current_type}"
            features[parent_child_key] = features.get(parent_child_key, 0) + 1
        
        # Extract keyword features within their syntactic context
        if hasattr(node, 'ttype') and node.ttype in Keyword:
            keyword = node.value.upper()
            features[f"KEYWORD:{keyword}"] = features.get(f"KEYWORD:{keyword}", 0) + 1
            
            # Contextual keyword features (within parent context)
            if parent_type is not None:
                features[f"KEYWORD_IN_{parent_type}:{keyword}"] = features.get(f"KEYWORD_IN_{parent_type}:{keyword}", 0) + 1
        
        # Extract function features
        if isinstance(node, Function):
            func_name = node.get_name()
            if func_name:
                features[f"FUNCTION:{func_name.upper()}"] = features.get(f"FUNCTION:{func_name.upper()}", 0) + 1
                
                # Contextual function features (within parent context)
                if parent_type is not None:
                    features[f"FUNCTION_IN_{parent_type}:{func_name.upper()}"] = features.get(f"FUNCTION_IN_{parent_type}:{func_name.upper()}", 0) + 1
        
        # Extract identifier features
        if isinstance(node, Identifier):
            name = node.get_real_name()
            if name:
                features[f"IDENTIFIER:{name.lower()}"] = features.get(f"IDENTIFIER:{name.lower()}", 0) + 1
                
                # Contextual identifier features (within parent context)
                if parent_type is not None:
                    features[f"IDENTIFIER_IN_{parent_type}:{name.lower()}"] = features.get(f"IDENTIFIER_IN_{parent_type}:{name.lower()}", 0) + 1
        
        # Recursively process child tokens
        if hasattr(node, 'tokens'):
            for token in node.tokens:
                traverse_ast(token, depth + 1, current_type)
    
    traverse_ast(parsed_query)
    return features

def compute_weighted_ast_similarity_v2(features1: Dict[str, int], features2: Dict[str, int], 
                                      feature_weights: Dict[str, float]) -> float:
    """
    Compute weighted similarity between two AST feature sets using v2 weights.
    
    Uses the original formula: Σ w(f) · min(c_Q_i(f), c_Q_j(f)) / Σ w(f)
    This provides meaningful differentiation between queries without forcing [0,1] range.
    """
    all_features = set(features1.keys()) | set(features2.keys())
    
    if not all_features:
        return 0.0
    
    total_similarity = 0.0
    total_weight = 0.0
    
    for feature in all_features:
        count1 = features1.get(feature, 0)
        count2 = features2.get(feature, 0)
        
        # Use minimum count as intersection
        intersection = min(count1, count2)
        
        # Get weight for this feature (default to 0.0 to avoid bias)
        weight = feature_weights.get(feature, 0.0)
        
        # Only add to totals if weight > 0 (learned features only)
        if weight > 0:
            total_similarity += intersection * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    similarity = total_similarity / total_weight
    return similarity

def main():
    """
    Main function to process all dev datasets using the v2 weighted similarity system.
    """
    # Dataset configurations with v2 weights files
    datasets = {
        'spider': {
            'dev_path': 'text2sql-datasets/spider/dev.json',
            'train_path': 'text2sql-datasets/spider/train_spider.json',
            'weights': 'learned-weights/spider_feature_importance_v2.json'
        },
        'sparc': {
            'dev_path': 'text2sql-datasets/sparc/dev.json',
            'train_path': 'text2sql-datasets/sparc/train.json',
            'weights': 'learned-weights/sparc_feature_importance_v2.json'
        },
        'cosql': {
            'dev_path': 'text2sql-datasets/cosql/sql_state_tracking/cosql_dev.json',
            'train_path': 'text2sql-datasets/cosql/sql_state_tracking/cosql_train.json',
            'weights': 'learned-weights/cosql_feature_importance_v2.json'
        }
    }
    
    output_dir = "similar_queries_results_v2"
    
    # Process each dataset
    for dataset_name, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} dataset with v2 weighted similarity")
        print(f"{'='*60}")
        
        # Debug: Check file existence
        print(f"Checking files for {dataset_name}:")
        print(f"  Dev path: {config['dev_path']} - {'EXISTS' if os.path.exists(config['dev_path']) else 'MISSING'}")
        print(f"  Train path: {config['train_path']} - {'EXISTS' if os.path.exists(config['train_path']) else 'MISSING'}")
        print(f"  Weights path: {config['weights']} - {'EXISTS' if os.path.exists(config['weights']) else 'MISSING'}")
        
        if not os.path.exists(config['dev_path']):
            print(f"Warning: {config['dev_path']} not found, skipping...")
            continue
            
        if not os.path.exists(config['train_path']):
            print(f"Warning: {config['train_path']} not found, skipping...")
            continue
            
        if not os.path.exists(config['weights']):
            print(f"Warning: {config['weights']} not found, skipping...")
            continue
        
        try:
            results = process_dev_dataset_v2(
                dataset_name, 
                config['dev_path'],
                config['train_path'],
                config['weights'], 
                output_dir
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("v2 Weighted similarity analysis complete!")
    print(f"Results saved in '{output_dir}' directory")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
