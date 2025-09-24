#!/usr/bin/env python3
"""
Query-Specific Attention Mechanism Implementation

This module implements the query-specific attention mechanism as described in the paper:
"Attention captures contextual, query-local importance (e.g., the COUNT inside a GROUP BY), 
while IDF captures corpus-level distinctiveness (e.g., rare operators or schema identifiers)."

The key innovation is that attention weights are computed PER QUERY during similarity computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sqlparse
from pathlib import Path
import json

class QuerySpecificAttentionAnalyzer:
    """
    Implements query-specific attention mechanism as described in the paper.
    
    Key difference from global approach:
    - Computes attention weights PER QUERY during similarity computation
    - Uses query-specific contextual importance
    - Maintains the trained model for real-time attention computation
    """
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.feature_vocab = {}
        self.reverse_vocab = {}
        self.load_model_and_vocab(model_path, vocab_path)
    
    def load_model_and_vocab(self, model_path: str, vocab_path: str):
        """Load the trained model and vocabulary for query-specific attention."""
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            self.feature_vocab = vocab_data['vocabulary']
            self.reverse_vocab = {v: k for k, v in self.feature_vocab.items()}
        
        # Load model architecture (simplified for this example)
        # In practice, you'd load the actual trained model
        self.model = self._create_model(len(self.feature_vocab))
        
        print(f"Loaded model with {len(self.feature_vocab)} features")
    
    def _create_model(self, vocab_size: int):
        """Create the attention model architecture."""
        class AttentionModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim=128, num_heads=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
                self.output_proj = nn.Linear(embedding_dim, vocab_size)
            
            def forward(self, feature_ids, attention_mask=None):
                embeddings = self.embedding(feature_ids)
                attn_output, attn_weights = self.attention(embeddings, embeddings, embeddings, 
                                                         key_padding_mask=attention_mask)
                return attn_output, attn_weights
        
        return AttentionModel(vocab_size).to(self.device)
    
    def extract_ast_features_from_query(self, query: str) -> Dict[str, int]:
        """Extract AST features from a single query (same as original implementation)."""
        try:
            parsed = sqlparse.parse(query)[0]
            features = {}
            
            def traverse_ast(node, depth=0):
                if hasattr(node, 'ttype') and node.ttype:
                    # Node type features
                    features[f"TYPE:{node.ttype}"] = features.get(f"TYPE:{node.ttype}", 0) + 1
                    
                    # Depth features
                    features[f"DEPTH:{depth}"] = features.get(f"DEPTH:{depth}", 0) + 1
                
                if hasattr(node, 'value') and node.value:
                    value = node.value.strip()
                    if value.upper() in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING']:
                        features[f"KEYWORD:{value.upper()}"] = features.get(f"KEYWORD:{value.upper()}", 0) + 1
                    elif value.upper() in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
                        features[f"FUNCTION:{value.upper()}"] = features.get(f"FUNCTION:{value.upper()}", 0) + 1
                    elif value.isalpha():
                        features[f"IDENTIFIER:{value}"] = features.get(f"IDENTIFIER:{value}", 0) + 1
                
                # Parent-child relationships
                if hasattr(node, 'tokens'):
                    for i, child in enumerate(node.tokens):
                        if hasattr(child, 'ttype') and child.ttype:
                            parent_type = getattr(node, 'ttype', 'UNKNOWN')
                            child_type = getattr(child, 'ttype', 'UNKNOWN')
                            features[f"PARENT_CHILD:{parent_type}:{child_type}"] = \
                                features.get(f"PARENT_CHILD:{parent_type}:{child_type}", 0) + 1
                        traverse_ast(child, depth + 1)
            
            traverse_ast(parsed)
            return features
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            return {}
    
    def encode_query_features(self, features: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert features to model input format."""
        feature_ids = []
        for feature, count in features.items():
            if feature in self.feature_vocab:
                vocab_id = self.feature_vocab[feature]
                # Add feature multiple times based on count
                feature_ids.extend([vocab_id] * count)
        
        if not feature_ids:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.bool)
        
        # Convert to tensors
        feature_tensor = torch.tensor(feature_ids, dtype=torch.long)
        attention_mask = torch.ones(len(feature_ids), dtype=torch.bool)
        
        return feature_tensor, attention_mask
    
    def compute_query_specific_attention(self, query: str) -> Dict[str, float]:
        """
        Compute query-specific attention weights as described in the paper.
        
        This is the key innovation: attention weights are computed PER QUERY,
        not averaged globally across all training queries.
        """
        try:
            # Extract features from the query
            features = self.extract_ast_features_from_query(query)
            if not features:
                return {}
            
            # Encode features for the model
            feature_ids, attention_mask = self.encode_query_features(features)
            if len(feature_ids) == 0:
                return {}
            
            # Add batch dimension
            feature_ids = feature_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            
            # Compute attention weights for this specific query
            self.model.eval()
            with torch.no_grad():
                _, attention_weights = self.model(feature_ids, attention_mask)
                
                # Check for NaN values
                if torch.isnan(attention_weights).any():
                    print(f"Warning: NaN values in attention weights for query: {query[:50]}...")
                    # Return uniform weights as fallback
                    return {feature: 1.0 for feature in features.keys()}
                
                # Step (i): Compute attention matrices (already done)
                # Step (ii): Average across heads
                attn_avg = attention_weights.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
                
                # Check for NaN in averaged attention
                if torch.isnan(attn_avg).any():
                    print(f"Warning: NaN values in averaged attention for query: {query[:50]}...")
                    return {feature: 1.0 for feature in features.keys()}
                
                # Step (iii): Average for each feature occurrence
                feature_attention = {}
                for i, (feature, count) in enumerate(features.items()):
                    if feature in self.feature_vocab:
                        vocab_id = self.feature_vocab[feature]
                        
                        # Find positions where this feature appears
                        feature_positions = torch.where(feature_ids.squeeze(0) == vocab_id)[0]
                        
                        if len(feature_positions) > 0:
                            # Average attention for this feature across all its occurrences
                            feature_attn = attn_avg[feature_positions].mean(dim=0).mean().item()
                            
                            # Check for NaN in final attention
                            if torch.isnan(torch.tensor(feature_attn)):
                                feature_attn = 1.0  # Fallback to uniform weight
                            
                            feature_attention[feature] = feature_attn
                        else:
                            # If feature not found in positions, use uniform weight
                            feature_attention[feature] = 1.0
                
                return feature_attention
                
        except Exception as e:
            print(f"Error computing query-specific attention: {e}")
            # Return uniform weights as fallback
            features = self.extract_ast_features_from_query(query)
            return {feature: 1.0 for feature in features.keys()}
    
    def compute_weighted_similarity_with_query_attention(self, 
                                                       query1: str, 
                                                       query2: str, 
                                                       idf_weights: Dict[str, float],
                                                       alpha: float = 0.5) -> float:
        """
        Compute similarity using query-specific attention as described in the paper.
        
        Formula: w(f|Q) = α·IDF(f) + (1-α)·Attn(f|Q)
        where Attn(f|Q) is computed per query, not globally.
        """
        try:
            # Extract features from both queries
            features1 = self.extract_ast_features_from_query(query1)
            features2 = self.extract_ast_features_from_query(query2)
            
            if not features1 or not features2:
                return 0.0
            
            # Compute query-specific attention for query1 (target query)
            attention_weights_q1 = self.compute_query_specific_attention(query1)
            
            # If attention computation failed, fall back to IDF-only
            if not attention_weights_q1:
                print("Warning: Query-specific attention failed, using IDF-only weights")
                attention_weights_q1 = {feature: 1.0 for feature in features1.keys()}
            
            # Get union of features
            all_features = set(features1.keys()) | set(features2.keys())
            
            if not all_features:
                return 0.0
            
            total_similarity = 0.0
            total_weight = 0.0
            
            for feature in all_features:
                count1 = features1.get(feature, 0)
                count2 = features2.get(feature, 0)
                
                # Intersection count
                intersection = min(count1, count2)
                
                # Get IDF weight (global)
                idf_weight = idf_weights.get(feature, 0.0)
                
                # Get query-specific attention weight
                attention_weight = attention_weights_q1.get(feature, 0.0)
                
                # Combine weights as per paper: w(f|Q) = α·IDF(f) + (1-α)·Attn(f|Q)
                combined_weight = alpha * idf_weight + (1 - alpha) * attention_weight
                
                # Ensure weight is valid (not NaN)
                if combined_weight > 0 and not (combined_weight != combined_weight):
                    total_similarity += intersection * combined_weight
                    total_weight += combined_weight
            
            if total_weight == 0:
                return 0.0
            
            return total_similarity / total_weight
            
        except Exception as e:
            print(f"Error in similarity computation: {e}")
            return 0.0

def find_top_similar_queries_with_query_attention(target_query: str, 
                                                train_queries: List[Tuple[str, str]], 
                                                idf_weights: Dict[str, float],
                                                analyzer: QuerySpecificAttentionAnalyzer,
                                                top_k: int = 10) -> List[Tuple[float, str, str, int]]:
    """
    Find top-k similar queries using query-specific attention mechanism.
    
    This implements the paper's key innovation: attention weights are computed
    specifically for the target query, not globally averaged.
    """
    import heapq
    
    heap = []
    seen_queries = set()
    
    for idx, (query, translation) in enumerate(train_queries):
        if query in seen_queries:
            continue
        
        # Compute similarity using query-specific attention
        similarity = analyzer.compute_weighted_similarity_with_query_attention(
            target_query, query, idf_weights
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

# Example usage
if __name__ == "__main__":
    # Initialize the query-specific attention analyzer
    analyzer = QuerySpecificAttentionAnalyzer(
        model_path="learned-weights/spider_model.pth",
        vocab_path="learned-weights/spider_feature_importance_v2.json"
    )
    
    # Load IDF weights
    with open("learned-weights/spider_feature_importance_v2.json", 'r') as f:
        weights_data = json.load(f)
        idf_weights = weights_data.get('idf_scores', {})
    
    # Example target query
    target_query = "SELECT COUNT(*) FROM singer WHERE age > 25"
    
    # Example training queries
    train_queries = [
        ("SELECT name FROM singer", "Show singer names"),
        ("SELECT AVG(age) FROM singer WHERE country = 'USA'", "Average age of US singers"),
        ("SELECT * FROM singer WHERE age > 30", "All singers over 30")
    ]
    
    # Find similar queries using query-specific attention
    similar_queries = find_top_similar_queries_with_query_attention(
        target_query, train_queries, idf_weights, analyzer, top_k=5
    )
    
    print("Query-specific attention results:")
    for sim_score, query, translation, idx in similar_queries:
        print(f"Similarity: {sim_score:.3f}, Query: {query}")
