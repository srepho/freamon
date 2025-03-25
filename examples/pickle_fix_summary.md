# Fixing AttributeError in Optimized Topic Modeling

## Issue

The `create_topic_model_optimized` function in `freamon/utils/text_utils.py` was encountering an error when attempting to pickle the model:

```
AttributeError: Can't pickle local object 'create_topic_model_optimized.<locals>.preprocess_batch'
```

This error occurred because the `preprocess_batch` function was defined locally within the `create_topic_model_optimized` function. Python cannot pickle local functions as they are bound to the local scope of their containing function.

## Solution

1. **Moved the `preprocess_batch` function outside of `create_topic_model_optimized`**:
   - Extracted the function definition and placed it at the module level
   - Added proper documentation for the function
   - Modified parameters to accept the processor and preprocessing options instead of accessing them from the enclosing scope

2. **Updated the function call sites**:
   - Changed the function call in the sequential processing section to pass the required arguments
   - Modified the multiprocessing call to use `pool.starmap()` with properly prepared arguments instead of `pool.imap()`

3. **Added tests to verify fix**:
   - Created a standalone test script to verify pickling functionality
   - Added a new test case to the existing test suite to ensure proper pickling support

## Implementation Details

### 1. New standalone preprocess_batch function:

```python
def preprocess_batch(batch_texts, processor, preproc_opts):
    """Process a batch of texts using specified preprocessing options.
    
    Args:
        batch_texts: List of texts to preprocess
        processor: TextProcessor instance
        preproc_opts: Dictionary of preprocessing options
        
    Returns:
        List of preprocessed texts
    """
    return [
        processor.preprocess_text(
            text, 
            remove_stopwords=preproc_opts['remove_stopwords'], 
            remove_punctuation=preproc_opts['remove_punctuation'],
            lemmatize=preproc_opts['use_lemmatization'],
            min_token_length=preproc_opts['min_token_length'],
            custom_stopwords=preproc_opts['custom_stopwords']
        ) for text in batch_texts
    ]
```

### 2. Updated multiprocessing call:

```python
# Create a list of arguments for the preprocess_batch function
preprocess_args = [(chunk, processor, preproc_opts) for chunk in chunks]

# Process chunks in parallel
print(f"Using {num_workers} workers for parallel preprocessing...")
with multiprocessing.Pool(processes=num_workers) as pool:
    results = []
    for i, result in enumerate(pool.starmap(preprocess_batch, preprocess_args)):
        results.append(result)
        # Report progress
        progress = min(100, (i + 1) * 100 // len(chunks))
        print(f"  Preprocessing progress: {progress}%", end='\r')
```

### 3. Impact and Benefits

- **Improved serialization**: The topic modeling results can now be properly pickled and unpickled
- **Better maintainability**: The extracted function is now more reusable and easier to understand
- **Test coverage**: Added explicit test for the pickle functionality

This fix ensures that models can be serialized, which is essential for:
- Saving trained models to disk
- Sharing models between processes
- Caching models for future use
- Deploying models in production environments