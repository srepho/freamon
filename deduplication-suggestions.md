# Adding Text Deduplication to Freamon

## Overview

This document outlines a plan for adding text deduplication functionality to the `freamon` package, leveraging the existing text processing capabilities. The implementation will support both exact and fuzzy deduplication methods, with a focus on practical use cases.

## Implementation Approach

Two primary approaches for implementation:

1. **Extend the existing `TextProcessor` class**: Add new methods for deduplication
2. **Create a new dedicated module**: For more complex deduplication workflows

## Extending the TextProcessor Class

Adding these methods to the existing `TextProcessor` class in `freamon/utils/text_utils.py`:

```python
def find_duplicates(
    self,
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    return_scores: bool = False
) -> Dict[int, List[Union[int, Tuple[int, float]]]]:
    """Find duplicate texts based on similarity."""

def deduplicate_texts(
    self,
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    keep: str = 'first'
) -> List[int]:
    """Return indices of unique texts after deduplication."""
```

## Creating a Dedicated Module

For more comprehensive deduplication capabilities, create a new module structure:

```
freamon/
└── deduplication/
    ├── __init__.py
    ├── fuzzy_deduplication.py
    ├── exact_deduplication.py
    ├── fingerprinting.py
    └── clustering.py
```

## Core Functionality

### 1. Exact Deduplication

- Hash-based deduplication (MD5, SHA)
- N-gram fingerprinting
- Case-sensitive and case-insensitive options

### 2. Fuzzy Deduplication

- Text similarity metrics:
  - Levenshtein distance
  - Jaccard similarity
  - Cosine similarity
- Embedding-based similarity (Word2Vec, GloVe, FastText)
- Locality-Sensitive Hashing (LSH) for efficient fuzzy matching

### 3. Document Fingerprinting

- Shingling
- MinHash
- SimHash

### 4. Clustering

- Hierarchical clustering
- DBSCAN clustering
- Connected components analysis

## Integration with Existing Code

The implementation will leverage existing functionality:

1. **Text Preprocessing**: Using existing methods from `TextProcessor`
2. **Document Similarity**: Building on `calculate_document_similarity` and `calculate_embedding_similarity`
3. **Embeddings**: Utilizing existing embedding creation methods

## Example Implementation

Here's a detailed implementation of a fuzzy deduplication method for the `TextProcessor` class:

```python
def find_duplicates(
    self,
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    return_scores: bool = False
) -> Dict[int, List[Union[int, Tuple[int, float]]]]:
    """
    Find duplicate texts based on similarity.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to check for duplicates.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    return_scores : bool, default=False
        Whether to include similarity scores in results.
    
    Returns
    -------
    Dict[int, List[Union[int, Tuple[int, float]]]]
        Dictionary mapping each text index to a list of duplicate indices 
        (with similarity scores if return_scores=True).
    """
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Preprocess texts if requested
    processed_texts = texts
    if preprocess:
        processed_texts = [
            self.preprocess_text(
                text,
                lowercase=True,
                remove_punctuation=True
            ) for text in texts
        ]
    
    # For embedding-based similarity
    if method == 'embedding':
        # Create document embeddings
        doc_embeddings = []
        
        # Check if word2vec model exists, otherwise create one
        if not hasattr(self, '_word_vectors'):
            embeddings = self.create_word2vec_embeddings(
                texts=processed_texts,
                vector_size=100
            )
            self._word_vectors = embeddings['wv']
        
        # Create document embeddings
        doc_embeddings = self.create_document_embeddings(
            texts=processed_texts,
            word_vectors=self._word_vectors,
            method='mean'
        )
        
        # Find duplicates using embeddings
        duplicates = {}
        for i in range(len(processed_texts)):
            duplicates[i] = []
            for j in range(len(processed_texts)):
                if i == j:
                    continue
                    
                similarity = self.calculate_embedding_similarity(
                    doc_embeddings[i], 
                    doc_embeddings[j],
                    method='cosine'
                )
                
                if similarity >= threshold:
                    if return_scores:
                        duplicates[i].append((j, similarity))
                    else:
                        duplicates[i].append(j)
        
        return duplicates
    
    # For text-based similarity
    duplicates = {}
    
    for i in range(len(processed_texts)):
        duplicates[i] = []
        
        for j in range(len(processed_texts)):
            if i == j:
                continue
                
            similarity = self.calculate_document_similarity(
                processed_texts[i],
                processed_texts[j],
                method=method
            )
            
            if similarity >= threshold:
                if return_scores:
                    duplicates[i].append((j, similarity))
                else:
                    duplicates[i].append(j)
    
    return duplicates
```

```python
def deduplicate_texts(
    self,
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    keep: str = 'first'
) -> List[int]:
    """
    Deduplicate texts based on similarity.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    
    Returns
    -------
    List[int]
        Indices of unique texts after deduplication.
    """
    # Find all duplicates
    duplicates_dict = self.find_duplicates(
        texts=texts,
        threshold=threshold,
        method=method,
        preprocess=preprocess,
        return_scores=True
    )
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Create a graph of similar texts
    import networkx as nx
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(len(texts)):
        G.add_node(i)
    
    # Add edges for duplicates
    for i, duplicates in duplicates_dict.items():
        for dup in duplicates:
            idx, score = dup if isinstance(dup, tuple) else (dup, 1.0)
            G.add_edge(i, idx, weight=score)
    
    # Find connected components (clusters of duplicates)
    clusters = list(nx.connected_components(G))
    
    # Choose which texts to keep from each cluster
    keep_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single text, keep it
            keep_indices.append(list(cluster)[0])
        else:
            # Multiple similar texts, choose based on keep strategy
            cluster_list = list(cluster)
            
            if keep == 'first':
                keep_idx = min(cluster_list)
            elif keep == 'last':
                keep_idx = max(cluster_list)
            elif keep == 'longest':
                keep_idx = max(cluster_list, key=lambda i: len(texts[i]))
            else:
                raise ValueError(f"Unknown keep strategy: {keep}. Use 'first', 'last', or 'longest'.")
            
            keep_indices.append(keep_idx)
    
    return sorted(keep_indices)
```

## Efficient Implementation for Large Datasets

For larger datasets, we can implement more efficient approaches:

```python
def lsh_deduplication(
    self,
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    preprocess: bool = True,
    keep: str = 'first'
) -> List[int]:
    """
    Deduplicate texts using Locality-Sensitive Hashing for scalability.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    num_perm : int, default=128
        Number of permutations for MinHash.
    bands : int, default=16
        Number of bands for LSH. More bands = higher recall, lower precision.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    
    Returns
    -------
    List[int]
        Indices of unique texts after deduplication.
    """
    try:
        import datasketch
    except ImportError:
        raise ImportError(
            "datasketch is not installed. Install it with 'pip install datasketch'."
        )
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Preprocess texts if requested
    processed_texts = texts
    if preprocess:
        processed_texts = [
            self.preprocess_text(
                text,
                lowercase=True,
                remove_punctuation=True
            ) for text in texts
        ]
    
    # Configure LSH parameters for the desired threshold
    # Estimate b and r values based on threshold
    rows = int(num_perm / bands)
    
    # Create MinHash LSH index
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # Create MinHashes for all texts
    minhashes = []
    for i, text in enumerate(processed_texts):
        m = datasketch.MinHash(num_perm=num_perm)
        for shingle in self._get_shingles(text):
            m.update(shingle.encode('utf8'))
        minhashes.append(m)
        lsh.insert(f"{i}", m)
    
    # Find duplicate clusters
    clusters = []
    processed = set()
    
    for i, m in enumerate(minhashes):
        if i in processed:
            continue
            
        # Find potential duplicates
        result = lsh.query(m)
        
        # Convert to indices
        indices = [int(idx) for idx in result]
        
        if len(indices) > 1:
            # Found a cluster
            clusters.append(set(indices))
            processed.update(indices)
        else:
            # Single item
            clusters.append({i})
            processed.add(i)
    
    # Choose which texts to keep from each cluster
    keep_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single text, keep it
            keep_indices.append(list(cluster)[0])
        else:
            # Multiple similar texts, choose based on keep strategy
            cluster_list = list(cluster)
            
            if keep == 'first':
                keep_idx = min(cluster_list)
            elif keep == 'last':
                keep_idx = max(cluster_list)
            elif keep == 'longest':
                keep_idx = max(cluster_list, key=lambda i: len(texts[i]))
            else:
                raise ValueError(f"Unknown keep strategy: {keep}. Use 'first', 'last', or 'longest'.")
            
            keep_indices.append(keep_idx)
    
    return sorted(keep_indices)

def _get_shingles(self, text: str, k: int = 3) -> List[str]:
    """
    Generate k-shingles from text.
    
    Parameters
    ----------
    text : str
        Input text.
    k : int, default=3
        Shingle size.
    
    Returns
    -------
    List[str]
        List of k-shingles.
    """
    words = text.split()
    
    # Return empty list for very short texts
    if len(words) < k:
        return []
    
    return [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]
```

## Integration with the Package

In `freamon/eda/__init__.py`, expose the new functionality:

```python
# Add to existing imports
from freamon.deduplication import (
    deduplicate_texts,
    find_duplicates,
    cluster_similar_texts
)

# Add to __all__
__all__ = [
    # Existing exports
    "EDAAnalyzer",
    "analyze_numeric",
    # ...
    # New deduplication functions
    "deduplicate_texts",
    "find_duplicates",
    "cluster_similar_texts"
]
```

## Technical Considerations

### 1. Scalability

For large datasets, consider these approaches:

- **MinHash and LSH**: Efficiently find similar items without pairwise comparison
- **Blocking techniques**: Group texts by common features before comparison
- **Parallelization**: Use multiprocessing for large-scale deduplication
- **Incremental processing**: Process data in chunks for memory efficiency

### 2. Memory Management

For very large text collections:

- **Streaming approaches**: Process one document at a time
- **Disk-based storage**: Use on-disk data structures for intermediate results
- **Feature hashing**: Reduce dimensionality of text representations

### 3. Evaluation Metrics

Metrics to evaluate deduplication quality:

- **Precision**: Fraction of identified duplicates that are actual duplicates
- **Recall**: Fraction of actual duplicates that were identified
- **F1 Score**: Harmonic mean of precision and recall
- **Runtime performance**: CPU time and memory usage

## Additional Features to Consider

1. **Interactive deduplication**: Allow user feedback for borderline cases
2. **Custom similarity metrics**: Let users define their own similarity functions
3. **Post-deduplication analysis**: Report on the types and quantities of duplicates found
4. **Integration with visualization**: Visualize duplicate clusters
5. **Document-level vs. paragraph-level deduplication**: Support different granularities

## Conclusion

Adding text deduplication to Freamon builds on its existing strengths in text analysis and will provide users with powerful tools for managing duplicate content. The implementation can be modular and scalable, allowing users to choose the appropriate method for their specific use case.
