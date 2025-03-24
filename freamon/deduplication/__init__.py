"""Text deduplication utilities for finding and removing duplicate texts."""

from typing import Dict, List, Tuple, Union, Optional, Any

from freamon.deduplication.exact_deduplication import (
    hash_deduplication,
    ngram_fingerprint_deduplication
)

from freamon.deduplication.fuzzy_deduplication import (
    find_similar_texts,
    deduplicate_texts,
    calculate_levenshtein_similarity,
    calculate_jaccard_similarity
)

from freamon.deduplication.fingerprinting import (
    create_shingled_document,
    create_minhash_signature,
    create_simhash_signature
)

from freamon.deduplication.clustering import (
    cluster_texts_hierarchical,
    cluster_texts_dbscan
)

from freamon.deduplication.lsh_deduplication import (
    lsh_deduplication,
    create_lsh_bands
)

__all__ = [
    # Exact deduplication
    'hash_deduplication',
    'ngram_fingerprint_deduplication',
    
    # Fuzzy deduplication
    'find_similar_texts',
    'deduplicate_texts',
    'calculate_levenshtein_similarity',
    'calculate_jaccard_similarity',
    
    # Document fingerprinting
    'create_shingled_document',
    'create_minhash_signature',
    'create_simhash_signature',
    
    # Clustering
    'cluster_texts_hierarchical',
    'cluster_texts_dbscan',
    
    # LSH deduplication
    'lsh_deduplication',
    'create_lsh_bands'
]
