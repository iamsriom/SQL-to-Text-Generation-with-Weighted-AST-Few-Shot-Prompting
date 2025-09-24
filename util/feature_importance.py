import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
from pathlib import Path
import sqlparse
from sqlparse.tokens import Keyword
from sqlparse.sql import Function, Identifier
import matplotlib.pyplot as plt

# --- DATASET LOADER ---
def load_all_queries(dataset_path: str, dataset_type: str) -> List[str]:
    """
    Load all SQL queries from a dataset, handling Spider, SParC, and CoSQL formats.
    Args:
        dataset_path: Path to the dataset JSON file.
        dataset_type: One of 'spider', 'sparc', 'cosql'.
    Returns:
        List of SQL query strings.
    """
    queries = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    if dataset_type == 'spider':
        for item in data:
            if 'query' in item:
                queries.append(item['query'])
    elif dataset_type == 'sparc':
        for item in data:
            for turn in item.get('interaction', []):
                if 'query' in turn:
                    queries.append(turn['query'])
    elif dataset_type == 'cosql':
        for item in data:
            # Add all interaction queries
            for turn in item.get('interaction', []):
                if 'query' in turn:
                    queries.append(turn['query'])
            # Add the final query if present
            if 'final' in item and 'query' in item['final']:
                queries.append(item['final']['query'])
    return queries

# --- SQL FEATURE EXTRACTOR ---
SQL_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN',
    'INNER JOIN', 'OUTER JOIN', 'ON', 'AS', 'AND', 'OR', 'UNION', 'INTERSECT', 'EXCEPT', 'LIMIT',
    'DISTINCT', 'IN', 'NOT', 'IS', 'NULL', 'BETWEEN', 'EXISTS', 'LIKE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
}

def extract_sql_features(query: str) -> List[str]:
    """
    Extract only SQL keywords/features from a query using proper multi-word keyword handling.
    """
    parsed = sqlparse.parse(query)
    if not parsed:
        return []
    
    # Get all keyword tokens
    tokens = [t for t in parsed[0].flatten() if t.ttype in Keyword]
    token_values = [t.value.upper() for t in tokens]
    
    features = []
    i = 0
    while i < len(token_values):
        # Check for multi-word keywords (2-word combinations)
        if i + 1 < len(token_values):
            two_word = f"{token_values[i]} {token_values[i+1]}"
            if two_word in SQL_KEYWORDS:
                features.append(two_word)
                i += 2  # Skip both tokens
                continue
        
        # Check for single-word keywords
        if token_values[i] in SQL_KEYWORDS:
            features.append(token_values[i])
        i += 1
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for f in features:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result

# --- IMPROVED MODEL CLASSES ---
class SQLFeatureEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        # vocab_size includes PAD token, but we need vocab_size + 1 for actual embedding
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)  # Reserve 0 for PAD
    
    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(feature_ids)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Mask PAD tokens in attention to prevent probability soaking
        if attention_mask is not None:
            # Expand mask for all heads: [batch_size, 1, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            # Apply mask: set PAD positions to large negative value
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply mask again to zero out attention weights for PAD tokens
        if attention_mask is not None:
            attention_weights = attention_weights * mask.float()
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.output(attended)
        return output, attention_weights

class SQLFeatureAnalyzer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.feature_embedding = SQLFeatureEmbedding(vocab_size, embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        # Main classification head (for continuity with existing code)
        self.classifier = nn.Linear(embedding_dim, 1)
        # NEW: BoW auxiliary head - predicts presence of every vocabulary feature (excluding PAD)
        self.bow_out = nn.Linear(embedding_dim, vocab_size)  # vocab_size is actual features (PAD excluded)
    
    def forward(self, feature_ids: torch.Tensor, attention_mask: torch.Tensor = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.feature_embedding(feature_ids)
        # Pass attention mask to attention layer for PAD masking
        attended_embeddings, attention_weights = self.attention(embeddings, attention_mask)
        
        if attention_mask is not None:
            masked_embeddings = attended_embeddings * attention_mask.unsqueeze(-1)
            pooled = torch.sum(masked_embeddings, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            pooled = torch.mean(attended_embeddings, dim=1)
        
        # Main classification head (scalar output)
        logits = self.classifier(pooled)
        
        # NEW: BoW head - predicts presence of each vocabulary feature
        bow_logits = self.bow_out(pooled)
        
        return logits, bow_logits, attended_embeddings, attention_weights

# --- IMPROVED MAIN ANALYZER ---
class FeatureImportanceAnalyzer:
    def __init__(self, output_dir: str = "learned-weights"):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.feature_vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.attention_scores = defaultdict(float)
        self.frequency_scores = defaultdict(float)
        self.combined_scores = defaultdict(float)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NEW: Loss weights for combined training
        self.lambda_bow = 1.0   # Weight for BoW loss (feature presence prediction) - primary objective
        self.lambda_cls = 0.0   # Weight for classification loss (disabled by default to avoid random noise)

    def extract_ast_features_from_query(self, query: str) -> Dict[str, int]:
        """
        Extract AST features from a SQL query with hierarchical relationships.
        Returns a dictionary of feature counts with namespaces: TYPE:, KEYWORD:, FUNCTION:, IDENTIFIER:, DEPTH:
        """
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return {}
            
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
            
            traverse_ast(parsed[0])
            return features
        except Exception as e:
            print(f"Error extracting AST features from query: {e}")
            return {}

    def build_multi_hot_target_from_encoded_ids(self, feature_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Build multi-hot target vector from encoded feature IDs to match truncation.
        This ensures BoW targets exactly match what the model sees after encoding.
        """
        target = torch.zeros(self.vocab_size, dtype=torch.float)
        
        # Only consider non-PAD positions
        valid_positions = torch.where(attention_mask > 0)[0]
        
        for pos in valid_positions:
            vocab_id = feature_ids[pos].item()
            if vocab_id > 0:  # Skip PAD token (index 0)
                # Map to BoW target indices (0 to vocab_size-1)
                bow_idx = vocab_id - 1  # Convert vocab index (1-based) to BoW index (0-based)
                if 0 <= bow_idx < self.vocab_size:
                    target[bow_idx] = 1.0
        
        return target

    def build_vocabulary(self, queries: List[str]) -> None:
        self.feature_vocab.clear()
        self.reverse_vocab.clear()
        all_features = set()
        for query in queries:
            # Use AST features instead of basic SQL keywords to match the paper description
            feats = self.extract_ast_features_from_query(query)
            all_features.update(feats.keys())
        
        # Start vocabulary at 1, reserve 0 for PAD token
        for i, feature in enumerate(sorted(all_features), start=1):
            self.feature_vocab[feature] = i
            self.reverse_vocab[i] = feature
        
        # Add PAD token explicitly
        self.feature_vocab['<PAD>'] = 0
        self.reverse_vocab[0] = '<PAD>'
        
        self.vocab_size = len(self.feature_vocab) - 1  # Exclude PAD from actual vocab size
        print(f"Built vocabulary with {self.vocab_size} AST features (PAD token reserved)")

    def encode_query(self, query: str, max_length: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:  # Increased token budget
        # Use AST features instead of basic SQL keywords
        ast_features = self.extract_ast_features_from_query(query)
        # Convert feature counts to list of features (repeat features based on their counts)
        features = []
        for feature, count in ast_features.items():
            features.extend([feature] * count)
        
        # Use PAD token (0) for padding, not a real feature
        feature_ids = [self.feature_vocab.get(f, 0) for f in features]  # 0 is now PAD token
        if len(feature_ids) > max_length:
            feature_ids = feature_ids[:max_length]
        else:
            feature_ids.extend([0] * (max_length - len(feature_ids)))  # Pad with 0 (PAD token)
        
        # Create attention mask: 1 for real features, 0 for PAD
        attention_mask = [1] * min(len(features), max_length) + [0] * max(0, max_length - len(features))
        
        return torch.tensor(feature_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.float)

    def compute_attention_importance(self, model: nn.Module, queries: List[str]) -> Dict[str, float]:
        model.eval()
        attention_scores = defaultdict(float)
        with torch.no_grad():
            for query in queries:
                feature_ids, attention_mask = self.encode_query(query)
                feature_ids = feature_ids.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)
                
                # Updated to handle new model output format
                _, _, _, attention_weights = model(feature_ids, attention_mask)
                
                # Get valid (non-PAD) positions and their corresponding vocab IDs
                mask = attention_mask.bool().squeeze(0)
                valid_positions = torch.where(mask)[0]
                valid_vocab_ids = feature_ids.squeeze(0)[valid_positions]
                
                if len(valid_positions) > 0:
                    # Average attention across heads and get attention for valid positions
                    attn = attention_weights.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
                    valid_attn = attn[valid_positions][:, valid_positions]  # [valid_len, valid_len]
                    
                    if valid_attn.numel() > 0:
                        # Get attention importance for each valid position
                        feature_importance = valid_attn.mean(dim=0)  # Average attention received by each position
                        
                        # Aggregate attention per feature ID to avoid double-counting repeated features
                        feature_attention_aggregated = defaultdict(float)
                        for i, importance in enumerate(feature_importance):
                            if i < len(valid_vocab_ids):
                                vocab_id = valid_vocab_ids[i].item()
                                if vocab_id != 0 and vocab_id in self.reverse_vocab:  # Skip PAD token
                                    feature_attention_aggregated[vocab_id] += importance.item()
                        
                        # Add aggregated attention to global scores
                        for vocab_id, aggregated_importance in feature_attention_aggregated.items():
                            feature = self.reverse_vocab[vocab_id]
                            attention_scores[feature] += aggregated_importance
        
        total_queries = len(queries)
        for feature in attention_scores:
            attention_scores[feature] /= total_queries
        
        # Normalize attention scores to [0,1]
        if attention_scores:
            max_attn = max(attention_scores.values())
            min_attn = min(attention_scores.values())
            if max_attn != min_attn:
                for feature in attention_scores:
                    attention_scores[feature] = (attention_scores[feature] - min_attn) / (max_attn - min_attn)
            else:
                # If all values are the same, set to 1.0
                for feature in attention_scores:
                    attention_scores[feature] = 1.0
        
        # Round to 3 decimal places
        for feature in attention_scores:
            attention_scores[feature] = round(attention_scores[feature], 3)
        
        return dict(attention_scores)

    def compute_frequency_importance(self, queries: List[str]) -> Dict[str, float]:
        feature_counts = Counter()
        total_queries = len(queries)
        for query in queries:
            features = extract_sql_features(query)
            feature_counts.update(features)
        frequency_scores = {}
        for feature, count in feature_counts.items():
            frequency_scores[feature] = count / total_queries
        return frequency_scores

    def compute_idf_importance(self, queries: List[str]) -> Dict[str, float]:
        """
        Compute inverse document frequency (IDF) for each AST feature.
        IDF(f) = log(N / (1 + count(f)))
        """
        feature_counts = Counter()
        total_queries = len(queries)
        for query in queries:
            # Use AST features instead of basic SQL keywords
            ast_features = self.extract_ast_features_from_query(query)
            features = set(ast_features.keys())  # Use set to count per-query presence
            feature_counts.update(features)
        idf_scores = {}
        import math
        for feature, count in feature_counts.items():
            idf_scores[feature] = math.log(total_queries / (1 + count))
        
        # Normalize IDF scores to [0,1]
        if idf_scores:
            max_idf = max(idf_scores.values())
            min_idf = min(idf_scores.values())
            if max_idf != min_idf:
                for feature in idf_scores:
                    idf_scores[feature] = (idf_scores[feature] - min_idf) / (max_idf - min_idf)
            else:
                # If all values are the same, set to 1.0
                for feature in idf_scores:
                    idf_scores[feature] = 1.0
        
        # Round to 3 decimal places
        for feature in idf_scores:
            idf_scores[feature] = round(idf_scores[feature], 3)
        
        return idf_scores

    def train_model(self, queries: List[str], labels: List[int], epochs: int = 5) -> None:
        """
        Train the improved model with combined BoW + classification loss.
        """
        if self.vocab_size == 0:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        self.model = SQLFeatureAnalyzer(self.vocab_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Pre-encode sequences and BoW targets once for efficiency
        encoded = []
        bow_targets = []
        for q in queries:
            feature_ids, attention_mask = self.encode_query(q)
            encoded.append((feature_ids, attention_mask))
            # Build BoW target from encoded IDs to match truncation
            bow_target = self.build_multi_hot_target_from_encoded_ids(feature_ids, attention_mask)
            bow_targets.append(bow_target)

        labels_tensor = torch.tensor(labels, dtype=torch.float).to(self.device)
        bow_targets = torch.stack(bow_targets).to(self.device)  # [N, vocab_size]

        # Loss functions
        bce_logits = nn.BCEWithLogitsLoss()       # for scalar classification head
        bce_bow = nn.BCEWithLogitsLoss()          # for multi-label BoW head

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for i, (feature_ids, attention_mask) in enumerate(encoded):
                feature_ids = feature_ids.unsqueeze(0).to(self.device)       # [1, L]
                attention_mask = attention_mask.unsqueeze(0).to(self.device) # [1, L]
                label = labels_tensor[i].unsqueeze(0)                        # [1]
                bow_target = bow_targets[i].unsqueeze(0)                     # [1, V]

                optimizer.zero_grad()
                
                # Model now returns: logits, bow_logits, attended_embeddings, attention_weights
                logits, bow_logits, _, _ = self.model(feature_ids, attention_mask)

                # Combined loss: BoW supervision + optional classification
                loss_cls = bce_logits(logits.view(-1), label)                # scalar label
                loss_bow = bce_bow(bow_logits, bow_target)                   # multi-label feature presence
                loss = self.lambda_bow * loss_bow + self.lambda_cls * loss_cls

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(queries):.4f}")
                print(f"  - BoW Loss: {self.lambda_bow:.1f} weight")
                print(f"  - Classification Loss: {self.lambda_cls:.1f} weight")

    def analyze_importance(self, queries: List[str], labels: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance using BoW supervision (primary) and optional classification labels.
        """
        self.build_vocabulary(queries)
        
        if labels is None:
            # Use BoW-only training (no random labels to avoid noise)
            print("Training with BoW supervision only (no classification labels)")
            self.train_model(queries, labels=[0] * len(queries))  # Dummy labels, BoW loss will dominate
        else:
            print(f"Training with BoW supervision + provided labels")
            self.train_model(queries, labels)
        
        print("Computing attention-based importance...")
        self.attention_scores = self.compute_attention_importance(self.model, queries)
        print("Computing IDF-based importance...")
        self.idf_scores = self.compute_idf_importance(queries)
        
        all_features = set(self.attention_scores.keys()) | set(self.idf_scores.keys())
        for feature in all_features:
            attn_score = self.attention_scores.get(feature, 0.0)
            idf_score = self.idf_scores.get(feature, 0.0)
            combined_score = 0.5 * attn_score + 0.5 * idf_score
            self.combined_scores[feature] = round(combined_score, 3)
        
        return {
            'attention': dict(self.attention_scores),
            'idf': dict(self.idf_scores),
            'combined': dict(self.combined_scores)
        }

    def analyze_importance_for_translation(self, queries: List[str], translations: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance specifically for SQL-to-text translation tasks.
        Uses translation quality as labels while BoW provides feature-level supervision.
        """
        print("Analyzing feature importance for translation tasks with BoW supervision...")
        print(f"Using {len(queries)} query-translation pairs")
        
        # Compute translation quality labels
        translation_labels = self._compute_translation_quality_labels(queries, translations)
        
        # Enable classification loss for translation task (small weight to avoid overwhelming BoW)
        original_lambda_cls = self.lambda_cls
        self.lambda_cls = 0.2  # Small weight for translation quality
        
        # Use translation quality labels with BoW supervision
        result = self.analyze_importance(queries, labels=translation_labels)
        
        # Restore original classification weight
        self.lambda_cls = original_lambda_cls
        
        print("Translation-aware analysis with BoW supervision complete!")
        print("Features are now weighted based on:")
        print("  1. Translation quality (via classification head)")
        print("  2. Feature presence patterns (via BoW head)")
        print("  3. Attention mechanisms (via multi-head attention)")
        
        return result

    def _compute_translation_quality_labels(self, queries: List[str], translations: List[str]) -> List[float]:
        """
        Compute translation quality labels based on semantic alignment between SQL and natural language.
        """
        labels = []
        for query, translation in zip(queries, translations):
            quality_score = 0.0
            
            # 1. Length ratio (balanced queries are easier to translate)
            sql_length = len(query.split())
            nl_length = len(translation.split())
            length_ratio = min(sql_length, nl_length) / max(sql_length, nl_length)
            quality_score += length_ratio * 0.3
            
            # 2. Complexity penalty (more complex SQL = harder translation)
            complexity = self._estimate_query_complexity(query)
            quality_score += (1.0 - complexity) * 0.4
            
            # 3. Keyword alignment (common SQL patterns = easier translation)
            alignment = self._compute_keyword_alignment(query, translation)
            quality_score += alignment * 0.3
            
            labels.append(min(1.0, max(0.0, quality_score)))
        
        return labels

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate SQL query complexity (0-1 scale)."""
        complexity = 0.0
        
        # Count complex constructs
        if 'JOIN' in query.upper():
            complexity += 0.3
        if 'SUBQUERY' in query.upper() or '(' in query and 'SELECT' in query[query.find('('):]:
            complexity += 0.4
        if 'GROUP BY' in query.upper():
            complexity += 0.2
        if 'HAVING' in query.upper():
            complexity += 0.2
        if 'UNION' in query.upper():
            complexity += 0.3
        if 'CASE' in query.upper():
            complexity += 0.2
        
        # Length factor
        word_count = len(query.split())
        if word_count > 50:
            complexity += 0.2
        elif word_count > 20:
            complexity += 0.1
            
        return min(1.0, complexity)

    def _compute_keyword_alignment(self, query: str, translation: str) -> float:
        """Compute how well SQL keywords align with natural language concepts."""
        sql_concepts = {
            'SELECT': ['find', 'get', 'show', 'list', 'display'],
            'FROM': ['from', 'in', 'using'],
            'WHERE': ['where', 'that', 'which', 'when'],
            'JOIN': ['with', 'and', 'together'],
            'COUNT': ['how many', 'number of', 'total'],
            'SUM': ['total', 'sum', 'add up'],
            'AVG': ['average', 'mean'],
            'MAX': ['highest', 'maximum', 'most'],
            'MIN': ['lowest', 'minimum', 'least'],
            'ORDER BY': ['sort', 'order', 'arrange'],
            'GROUP BY': ['group', 'categorize', 'by']
        }
        
        query_upper = query.upper()
        translation_lower = translation.lower()
        
        alignment_score = 0.0
        total_keywords = 0
        
        for keyword, concepts in sql_concepts.items():
            if keyword in query_upper:
                total_keywords += 1
                if any(concept in translation_lower for concept in concepts):
                    alignment_score += 1.0
        
        return alignment_score / max(1, total_keywords)

    def save_weights(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        weights_data = {
            'dataset_name': dataset_name,
            'vocabulary': self.feature_vocab,
            'attention_scores': dict(self.attention_scores),
            'idf_scores': dict(self.idf_scores),
            'combined_scores': dict(self.combined_scores),
            'vocab_size': self.vocab_size,
            'total_features_analyzed': len(self.combined_scores),
            'training_method': 'BoW_auxiliary_head',
            'loss_weights': {
                'lambda_bow': self.lambda_bow,
                'lambda_cls': self.lambda_cls
            }
        }
        if dataset_name:
            output_path = self.output_dir / f"{dataset_name}_feature_importance_v2.json"
            with open(output_path, 'w') as f:
                json.dump(weights_data, f, indent=2)
            print(f"Saved {dataset_name} weights (v2 with BoW) to {output_path}")
        return weights_data

    def visualize_importance(self, top_k: int = 20) -> None:
        """Create visualizations of feature importance."""
        # Sort features by combined importance
        sorted_features = sorted(self.combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Combined Importance Score (BoW + Attention + IDF)')
        plt.title('Top SQL Features by Importance (Improved with BoW Supervision)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "feature_importance_v2_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {plot_path}")
        
        # Create detailed comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Attention scores
        attn_sorted = sorted(self.attention_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        attn_features, attn_scores = zip(*attn_sorted)
        axes[0].barh(range(len(attn_features)), attn_scores)
        axes[0].set_title('Attention-based Importance (BoW-supervised)')
        axes[0].set_yticks(range(len(attn_features)))
        axes[0].set_yticklabels(attn_features)
        
        # IDF scores
        idf_sorted = sorted(self.idf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        idf_features, idf_scores = zip(*idf_sorted)
        axes[1].barh(range(len(idf_features)), idf_scores)
        axes[1].set_title('IDF-based Importance')
        axes[1].set_yticks(range(len(idf_features)))
        axes[1].set_yticklabels(idf_features)
        
        # Combined scores
        axes[2].barh(range(len(features)), scores)
        axes[2].set_title('Combined Importance (BoW + Attention + IDF)')
        axes[2].set_yticks(range(len(features)))
        axes[2].set_yticklabels(features)
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_plot_path = self.output_dir / "detailed_importance_v2_plot.png"
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved detailed visualization to {detailed_plot_path}")
