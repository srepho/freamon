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

from freamon.deduplication.supervised_deduplication import (
    SupervisedDeduplicationModel
)

from freamon.deduplication.report import (
    generate_deduplication_report,
    export_deduplication_report,
    prepare_duplicate_report_data
)

# Import duplicate flagging functionality
from freamon.deduplication.flag_duplicates import (
    flag_exact_duplicates,
    flag_text_duplicates,
    flag_similar_records,
    flag_supervised_duplicates,
    add_duplicate_detection_columns
)

# Import Polars-optimized implementations
from freamon.deduplication.polars_lsh_deduplication import (
    polars_lsh_deduplication,
    streaming_lsh_deduplication,
    batch_process_texts,
    batch_create_minhash_signatures
)

from freamon.deduplication.polars_supervised_deduplication import (
    PolarsSupervisedDeduplicationModel
)

# Import evaluation functionality if available
try:
    from freamon.deduplication.evaluation import (
        calculate_deduplication_metrics,
        plot_confusion_matrix,
        evaluate_threshold_sensitivity,
        generate_evaluation_report,
        flag_and_evaluate
    )
    has_evaluation = True
except ImportError:
    has_evaluation = False

# Import advanced LSH functionality if available
try:
    from freamon.deduplication.advanced_lsh import (
        SimHash,
        BKTree,
        SuperMinHash,
        flag_similar_records_advanced_lsh,
        compare_lsh_methods
    )
    has_advanced_lsh = True
except ImportError:
    has_advanced_lsh = False

# Import enhanced reporting functionality if available
try:
    from freamon.deduplication.enhanced_reporting import (
        EnhancedDeduplicationReport,
        generate_enhanced_report,
        display_jupyter_report
    )
    has_enhanced_reporting = True
except ImportError:
    has_enhanced_reporting = False

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
    'create_lsh_bands',
    
    # Supervised deduplication
    'SupervisedDeduplicationModel',
    
    # Deduplication reporting
    'generate_deduplication_report',
    'export_deduplication_report',
    'prepare_duplicate_report_data',
    
    # Duplicate flagging functionality
    'flag_exact_duplicates',
    'flag_text_duplicates',
    'flag_similar_records',
    'flag_supervised_duplicates',
    'add_duplicate_detection_columns',
    
    # Polars-optimized LSH deduplication
    'polars_lsh_deduplication',
    'streaming_lsh_deduplication',
    'batch_process_texts',
    'batch_create_minhash_signatures',
    
    # Polars-optimized supervised deduplication
    'PolarsSupervisedDeduplicationModel',
]

# Add evaluation functions to __all__ if available
if has_evaluation:
    __all__.extend([
        # Evaluation functionality
        'calculate_deduplication_metrics',
        'plot_confusion_matrix',
        'evaluate_threshold_sensitivity',
        'generate_evaluation_report',
        'flag_and_evaluate'
    ])

# Add advanced LSH functions to __all__ if available
if has_advanced_lsh:
    __all__.extend([
        # Advanced LSH functionality
        'SimHash',
        'BKTree',
        'SuperMinHash',
        'flag_similar_records_advanced_lsh',
        'compare_lsh_methods'
    ])

# Add enhanced reporting functions to __all__ if available
if has_enhanced_reporting:
    __all__.extend([
        # Enhanced reporting functionality
        'EnhancedDeduplicationReport',
        'generate_enhanced_report',
        'display_jupyter_report'
    ])
