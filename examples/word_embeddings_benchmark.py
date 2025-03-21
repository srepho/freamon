"""
Benchmark for word embedding capabilities in freamon.
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

from freamon.utils.text_utils import TextProcessor

# Constants for benchmarking
MAX_DOCS = 1000
EMBEDDING_TYPES = ['word2vec', 'fasttext', 'glove']
DIMENSIONS = [50, 100, 300]
AGGREGATION_METHODS = ['mean', 'idf']
ITERATIONS = 3  # Number of runs for stable measurements

def load_dataset(max_docs=1000):
    """Load the 20 newsgroups dataset for benchmarking."""
    print(f"Loading up to {max_docs} documents from 20 newsgroups dataset...")
    categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Limit to max_docs
    data = newsgroups.data[:max_docs]
    targets = newsgroups.target[:max_docs]
    
    df = pd.DataFrame({
        'text': data,
        'category': [newsgroups.target_names[target] for target in targets]
    })
    
    return df

def benchmark_training(processor, texts, vector_sizes=[50, 100], iterations=3):
    """Benchmark Word2Vec training with different parameters."""
    training_results = []
    
    for vector_size in vector_sizes:
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            word2vec = processor.create_word2vec_embeddings(
                texts=texts,
                vector_size=vector_size,
                window=5,
                min_count=5,
                epochs=5,
                seed=42
            )
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
        training_results.append({
            'vector_size': vector_size,
            'mean_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'vocab_size': word2vec['vocab_size'],
        })
    
    return training_results

def benchmark_embedding_loading(processor, vector_sizes=[50, 100], iterations=3):
    """Benchmark loading pretrained embeddings."""
    loading_results = []
    
    for embedding_type in ['glove', 'fasttext']:
        for dimension in vector_sizes:
            times = []
            for _ in range(iterations):
                try:
                    start_time = time.time()
                    
                    embeddings = processor.load_pretrained_embeddings(
                        embedding_type=embedding_type,
                        dimension=dimension,
                        limit=10000,  # Limit vocabulary size for benchmarking
                        offline_mode=True  # First try with offline mode
                    )
                    
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    
                except Exception as e:
                    print(f"Error loading {embedding_type} {dimension}d: {str(e)}")
                    times = [float('nan')]
                    break
            
            if not any(np.isnan(times)):
                loading_results.append({
                    'embedding_type': embedding_type,
                    'dimension': dimension,
                    'mean_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'vocab_size': embeddings.get('vocab_size', 0) if 'embeddings' in locals() else 0
                })
    
    return loading_results

def benchmark_document_embedding(processor, texts, word_vectors, methods=['mean', 'idf'], iterations=3):
    """Benchmark creating document embeddings with different methods."""
    embedding_results = []
    
    for method in methods:
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            doc_embeddings = processor.create_document_embeddings(
                texts=texts,
                word_vectors=word_vectors,
                method=method
            )
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        embedding_results.append({
            'method': method,
            'mean_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'doc_count': len(texts),
            'vector_size': word_vectors.vector_size
        })
    
    return embedding_results

def benchmark_classification(processor, df, embedding_type='word2vec', vector_size=100):
    """Benchmark document classification using embeddings."""
    X = df['text']
    y = df['category']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create Word2Vec embeddings
    word2vec = processor.create_word2vec_embeddings(
        texts=X_train,
        vector_size=vector_size,
        window=5,
        min_count=2,
        epochs=5,
        seed=42
    )
    
    # Create document embeddings
    start_time = time.time()
    train_embeddings = processor.create_document_embeddings(
        texts=X_train,
        word_vectors=word2vec['wv'],
        method='mean'
    )
    test_embeddings = processor.create_document_embeddings(
        texts=X_test,
        word_vectors=word2vec['wv'],
        method='mean'
    )
    embedding_time = time.time() - start_time
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(20, vector_size))
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)
    
    # Train a classifier
    start_time = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings_pca, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    y_pred = clf.predict(test_embeddings_pca)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'embedding_type': embedding_type,
        'vector_size': vector_size,
        'embedding_time': embedding_time,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'accuracy': accuracy,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

def benchmark_similarity_calculation(processor, iterations=100):
    """Benchmark similarity calculation methods."""
    # Create random vectors
    vector_size = 100
    vectors = np.random.random((1000, vector_size))
    
    similarity_results = []
    for method in ['cosine', 'euclidean', 'dot']:
        start_time = time.time()
        
        # Calculate pairwise similarities
        for _ in range(iterations):
            i = np.random.randint(0, len(vectors))
            j = np.random.randint(0, len(vectors))
            processor.calculate_embedding_similarity(
                vectors[i], vectors[j], method=method
            )
        
        elapsed = time.time() - start_time
        similarity_results.append({
            'method': method,
            'time': elapsed,
            'iterations': iterations,
            'time_per_calculation': elapsed / iterations
        })
    
    return similarity_results

# Main benchmark function
def run_benchmarks():
    """Run all word embedding benchmarks."""
    print("Starting Word Embedding Benchmarks")
    print("-" * 50)
    
    # Initialize TextProcessor
    processor = TextProcessor(use_spacy=False)
    
    # Load dataset
    df = load_dataset(max_docs=MAX_DOCS)
    print(f"Loaded {len(df)} documents")
    
    # Benchmark 1: Word2Vec Training
    print("\nBenchmark 1: Word2Vec Training")
    print("-" * 50)
    training_results = benchmark_training(processor, df['text'], vector_sizes=DIMENSIONS[:2])
    
    print("Results:")
    for result in training_results:
        print(f"  Vector Size: {result['vector_size']}, Mean Time: {result['mean_time']:.2f}s, "  
              f"Vocab Size: {result['vocab_size']}")
    
    # Benchmark 2: Document Embedding Creation
    print("\nBenchmark 2: Document Embedding Creation")
    print("-" * 50)
    
    # First create Word2Vec model to use for document embeddings
    word2vec = processor.create_word2vec_embeddings(
        texts=df['text'],
        vector_size=100,
        window=5,
        min_count=2,
        epochs=5,
        seed=42
    )
    
    embedding_results = benchmark_document_embedding(
        processor, df['text'], word2vec['wv'], methods=AGGREGATION_METHODS
    )
    
    print("Results:")
    for result in embedding_results:
        print(f"  Method: {result['method']}, Mean Time: {result['mean_time']:.2f}s, "  
              f"Documents: {result['doc_count']}")
    
    # Benchmark 3: Similarity Calculation
    print("\nBenchmark 3: Similarity Calculation")
    print("-" * 50)
    similarity_results = benchmark_similarity_calculation(processor, iterations=1000)
    
    print("Results:")
    for result in similarity_results:
        print(f"  Method: {result['method']}, Total Time: {result['time']:.4f}s, "  
              f"Time per calculation: {result['time_per_calculation']*1000:.4f}ms")
    
    # Benchmark 4: Classification Accuracy
    print("\nBenchmark 4: Classification Accuracy")
    print("-" * 50)
    
    classification_results = []
    for vector_size in [50, 100]:
        result = benchmark_classification(processor, df, vector_size=vector_size)
        classification_results.append(result)
        print(f"  Vector Size: {vector_size}, Accuracy: {result['accuracy']:.4f}, "  
              f"Embedding Time: {result['embedding_time']:.2f}s")
    
    # Plot results
    print("\nGenerating benchmark plots...")
    
    # Plot 1: Training time vs. vector size
    plt.figure(figsize=(10, 6))
    plt.bar(
        [str(r['vector_size']) for r in training_results],
        [r['mean_time'] for r in training_results]
    )
    plt.xlabel('Vector Size')
    plt.ylabel('Training Time (s)')
    plt.title('Word2Vec Training Time vs. Vector Size')
    plt.savefig('benchmark_training_time.png')
    
    # Plot 2: Document embedding time by method
    plt.figure(figsize=(10, 6))
    plt.bar(
        [r['method'] for r in embedding_results],
        [r['mean_time'] for r in embedding_results]
    )
    plt.xlabel('Aggregation Method')
    plt.ylabel('Embedding Time (s)')
    plt.title('Document Embedding Time by Method')
    plt.savefig('benchmark_embedding_time.png')
    
    # Plot 3: Classification accuracy by vector size
    plt.figure(figsize=(10, 6))
    plt.bar(
        [str(r['vector_size']) for r in classification_results],
        [r['accuracy'] for r in classification_results]
    )
    plt.xlabel('Vector Size')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Vector Size')
    plt.savefig('benchmark_classification_accuracy.png')
    
    print("Benchmark plots saved")
    print("Benchmarks complete!")

if __name__ == "__main__":
    run_benchmarks()