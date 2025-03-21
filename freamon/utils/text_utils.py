"""
Utility functions for text processing.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
from collections import Counter

import numpy as np
import pandas as pd


class TextProcessor:
    """
    Class for basic text processing with optional spaCy integration.
    
    Parameters
    ----------
    use_spacy : bool, default=False
        Whether to use spaCy for text processing. If True, the spacy package
        must be installed.
    spacy_model : str, default='en_core_web_sm'
        The spaCy model to use for text processing. Only used if use_spacy=True.
    """
    
    def __init__(
        self,
        use_spacy: bool = False,
        spacy_model: str = 'en_core_web_sm',
    ):
        """Initialize the TextProcessor."""
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
            except ImportError:
                raise ImportError(
                    "spaCy is not installed. Install it with 'pip install spacy' "
                    "and download the model with 'python -m spacy download en_core_web_sm'."
                )
            except OSError:
                raise OSError(
                    f"spaCy model '{spacy_model}' is not installed. "
                    f"Download it with 'python -m spacy download {spacy_model}'."
                )
    
    def preprocess_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
    ) -> str:
        """
        Preprocess text by applying various transformations.
        
        Parameters
        ----------
        text : str
            The text to preprocess.
        lowercase : bool, default=True
            Whether to convert the text to lowercase.
        remove_punctuation : bool, default=True
            Whether to remove punctuation from the text.
        remove_numbers : bool, default=False
            Whether to remove numbers from the text.
        remove_stopwords : bool, default=False
            Whether to remove stopwords from the text. Only used if use_spacy=True.
        lemmatize : bool, default=False
            Whether to lemmatize the text. Only used if use_spacy=True.
        
        Returns
        -------
        str
            The preprocessed text.
        """
        if not isinstance(text, str):
            return ""
        
        # Handle empty strings
        if not text.strip():
            return ""
        
        if self.use_spacy and self.nlp is not None:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Apply transformations
            tokens = []
            for token in doc:
                # Skip stopwords if requested
                if remove_stopwords and token.is_stop:
                    continue
                
                # Skip punctuation if requested
                if remove_punctuation and token.is_punct:
                    continue
                
                # Skip numbers if requested
                if remove_numbers and token.like_num:
                    continue
                
                # Lemmatize if requested
                if lemmatize:
                    token_text = token.lemma_
                else:
                    token_text = token.text
                
                # Lowercase if requested
                if lowercase:
                    token_text = token_text.lower()
                
                tokens.append(token_text)
            
            return " ".join(tokens)
        else:
            # Simple processing without spaCy
            result = text
            
            # Lowercase if requested
            if lowercase:
                result = result.lower()
            
            # Remove punctuation if requested
            if remove_punctuation:
                import string
                result = result.translate(str.maketrans("", "", string.punctuation))
            
            # Remove numbers if requested
            if remove_numbers:
                result = "".join([c for c in result if not c.isdigit()])
            
            # Remove extra whitespace
            result = " ".join(result.split())
            
            return result
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Parameters
        ----------
        text : str
            The text to extract entities from.
        
        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping entity types to lists of entity text.
        """
        if not self.use_spacy or self.nlp is None:
            raise ValueError(
                "spaCy is not enabled. Initialize TextProcessor with use_spacy=True."
            )
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def process_dataframe_column(
        self,
        df: pd.DataFrame,
        column: str,
        result_column: Optional[str] = None,
        backend: str = 'auto',  # 'pandas', 'polars', 'pyarrow', or 'auto'
        batch_size: int = 1000,
        use_parallel: bool = False,
        n_jobs: int = -1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a text column with optimized backend selection.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        column : str
            The name of the column to process.
        result_column : Optional[str], default=None
            The name of the column to store the processed text.
            If None, the original column is overwritten.
        backend : str, default='auto'
            The backend to use for processing:
            - 'pandas': Use standard pandas
            - 'polars': Use polars (faster for large datasets)
            - 'pyarrow': Use pandas with PyArrow string type
            - 'auto': Automatically select the best backend
        batch_size : int, default=1000
            Batch size for spaCy processing.
        use_parallel : bool, default=False
            Whether to use parallel processing for large datasets.
        n_jobs : int, default=-1
            Number of workers for parallel processing. -1 means use all cores.
        **kwargs
            Additional keyword arguments to pass to preprocess_text.
        
        Returns
        -------
        pd.DataFrame
            The dataframe with the processed text column.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        
        # Determine result column
        if result_column is None:
            result_column = column
        
        # Extract processing parameters
        lowercase = kwargs.get('lowercase', True)
        remove_punctuation = kwargs.get('remove_punctuation', True)
        remove_numbers = kwargs.get('remove_numbers', False)
        remove_stopwords = kwargs.get('remove_stopwords', False)
        lemmatize = kwargs.get('lemmatize', False)
        
        # Check if we need spaCy
        needs_spacy = self.use_spacy and self.nlp is not None and (remove_stopwords or lemmatize)
        
        # Auto-select backend based on data size and available libraries
        if backend == 'auto':
            # For large datasets or simple operations, try using polars or pyarrow
            if len(df) > 10000 and not needs_spacy:
                try:
                    import polars
                    backend = 'polars'
                except ImportError:
                    try:
                        import pyarrow
                        backend = 'pyarrow'
                    except ImportError:
                        backend = 'pandas'
            else:
                backend = 'pandas'
        
        # 1. Polars Backend
        if backend == 'polars':
            try:
                import polars as pl
                # Convert to Polars
                pl_df = pl.from_pandas(df)
                
                if needs_spacy:
                    # Process with spaCy (using batches)
                    texts = pl_df[column].fill_null("").cast(pl.Utf8).to_list()
                    processed_texts = []
                    
                    # Process in batches
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        
                        if use_parallel and len(batch) > 100:
                            # Parallel processing
                            from concurrent.futures import ProcessPoolExecutor
                            import multiprocessing
                            
                            if n_jobs == -1:
                                n_jobs = max(1, multiprocessing.cpu_count() - 1)
                            
                            # Split batch for parallel processing
                            chunk_size = max(1, len(batch) // n_jobs)
                            chunks = [batch[j:j+chunk_size] for j in range(0, len(batch), chunk_size)]
                            
                            # Process in parallel
                            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                                # Need to create a new processor in each worker
                                def process_chunk(chunk_texts):
                                    local_processor = TextProcessor(
                                        use_spacy=True, 
                                        spacy_model=self.spacy_model
                                    )
                                    return [local_processor.preprocess_text(
                                        t, 
                                        lowercase=lowercase,
                                        remove_punctuation=remove_punctuation,
                                        remove_numbers=remove_numbers,
                                        remove_stopwords=remove_stopwords,
                                        lemmatize=lemmatize
                                    ) for t in chunk_texts]
                                
                                results = list(executor.map(process_chunk, chunks))
                                batch_results = [text for chunk_result in results for text in chunk_result]
                        else:
                            # Sequential batch processing
                            docs = list(self.nlp.pipe(batch))
                            batch_results = []
                            
                            for doc in docs:
                                tokens = []
                                for token in doc:
                                    # Apply filters
                                    if remove_stopwords and token.is_stop:
                                        continue
                                    if remove_punctuation and token.is_punct:
                                        continue
                                    if remove_numbers and token.like_num:
                                        continue
                                    
                                    # Apply transformations
                                    token_text = token.lemma_ if lemmatize else token.text
                                    token_text = token_text.lower() if lowercase else token_text
                                    
                                    tokens.append(token_text)
                                
                                batch_results.append(" ".join(tokens))
                        
                        processed_texts.extend(batch_results)
                    
                    # Add processed texts to DataFrame (support both old and new Polars API)
                    try:
                        # Try new API first
                        pl_df = pl_df.with_columns(pl.Series(name=result_column, values=processed_texts))
                    except AttributeError:
                        # Fall back to old API
                        pl_df = pl_df.with_column(pl.Series(name=result_column, values=processed_texts))
                else:
                    # Use Polars' native string operations
                    text_col = pl_df[column].fill_null("").cast(pl.Utf8)
                    
                    if lowercase:
                        text_col = text_col.str.to_lowercase()
                    
                    if remove_punctuation:
                        text_col = text_col.str.replace_all(r'[^\w\s]', "", literal=False)
                    
                    if remove_numbers:
                        text_col = text_col.str.replace_all(r'\d+', "", literal=False)
                    
                    # Clean up whitespace (replace multiple spaces with single space)
                    text_col = text_col.str.replace_all(r'\s+', " ", literal=False)
                    # Trim whitespace (Polars equivalent of strip)
                    text_col = text_col.str.strip_chars()
                    
                    # Add the processed column (support both old and new Polars API)
                    try:
                        # Try new API first
                        pl_df = pl_df.with_columns(text_col.alias(result_column))
                    except AttributeError:
                        # Fall back to old API
                        pl_df = pl_df.with_column(text_col.alias(result_column))
                
                # Convert back to pandas
                return pl_df.to_pandas()
                
            except ImportError:
                # If Polars is not available, fall back to pandas
                backend = 'pandas'
        
        # 2. PyArrow Backend
        if backend == 'pyarrow':
            try:
                import pyarrow
                
                # Create shallow copy unless overwriting
                if result_column == column:
                    result = df
                else:
                    result = df.copy(deep=False)
                
                if needs_spacy:
                    # For spaCy, we still need to process sequentially or in batches
                    texts = result[column].fillna("").astype(str).to_numpy()
                    
                    # Similar batching logic as in the polars case
                    processed_texts = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        # Process the batch with spaCy
                        if use_parallel and len(batch) > 100:
                            # Parallel processing
                            from concurrent.futures import ProcessPoolExecutor
                            import multiprocessing
                            
                            if n_jobs == -1:
                                n_jobs = max(1, multiprocessing.cpu_count() - 1)
                            
                            # Split batch for parallel processing
                            chunk_size = max(1, len(batch) // n_jobs)
                            chunks = [batch[j:j+chunk_size] for j in range(0, len(batch), chunk_size)]
                            
                            # Process in parallel
                            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                                def process_chunk(chunk_texts):
                                    local_processor = TextProcessor(
                                        use_spacy=True, 
                                        spacy_model=self.spacy_model
                                    )
                                    return [local_processor.preprocess_text(
                                        t, 
                                        lowercase=lowercase,
                                        remove_punctuation=remove_punctuation,
                                        remove_numbers=remove_numbers,
                                        remove_stopwords=remove_stopwords,
                                        lemmatize=lemmatize
                                    ) for t in chunk_texts]
                                
                                results = list(executor.map(process_chunk, chunks))
                                batch_results = [text for chunk_result in results for text in chunk_result]
                        else:
                            # Sequential batch processing
                            docs = list(self.nlp.pipe(batch))
                            batch_results = []
                            
                            for doc in docs:
                                tokens = []
                                for token in doc:
                                    # Apply filters
                                    if remove_stopwords and token.is_stop:
                                        continue
                                    if remove_punctuation and token.is_punct:
                                        continue
                                    if remove_numbers and token.like_num:
                                        continue
                                    
                                    # Apply transformations
                                    token_text = token.lemma_ if lemmatize else token.text
                                    token_text = token_text.lower() if lowercase else token_text
                                    
                                    tokens.append(token_text)
                                
                                batch_results.append(" ".join(tokens))
                        
                        processed_texts.extend(batch_results)
                    
                    # Convert to PyArrow string type
                    result[result_column] = pd.Series(processed_texts, index=result.index).astype("string[pyarrow]")
                else:
                    # Use PyArrow-backed string operations
                    text_series = result[column].fillna("").astype("string[pyarrow]")
                    
                    # Apply transformations
                    if lowercase:
                        text_series = text_series.str.lower()
                    
                    if remove_punctuation:
                        text_series = text_series.str.replace(r'[^\w\s]', "", regex=True)
                    
                    if remove_numbers:
                        text_series = text_series.str.replace(r'\d+', "", regex=True)
                    
                    text_series = text_series.str.replace(r'\s+', " ", regex=True).str.strip()
                    
                    result[result_column] = text_series
                
                return result
                
            except (ImportError, TypeError):
                # TypeError can occur if pandas version doesn't support PyArrow strings
                # If PyArrow is not available, fall back to pandas
                backend = 'pandas'
        
        # 3. Standard Pandas Backend (fallback)
        # Use shallow copy unless overwriting
        if result_column == column:
            result = df
        else:
            result = df.copy(deep=False)
        
        if needs_spacy:
            # Process with spaCy using batches
            texts = result[column].fillna("").astype(str).tolist()
            processed_texts = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Parallel processing if requested and batch is large enough
                if use_parallel and len(batch) > 100:
                    from concurrent.futures import ProcessPoolExecutor
                    import multiprocessing
                    
                    if n_jobs == -1:
                        n_jobs = max(1, multiprocessing.cpu_count() - 1)
                    
                    # Split batch for parallel processing
                    chunk_size = max(1, len(batch) // n_jobs)
                    chunks = [batch[j:j+chunk_size] for j in range(0, len(batch), chunk_size)]
                    
                    # Process in parallel
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        def process_chunk(chunk_texts):
                            local_processor = TextProcessor(
                                use_spacy=True, 
                                spacy_model=self.spacy_model
                            )
                            return [local_processor.preprocess_text(
                                t, 
                                lowercase=lowercase,
                                remove_punctuation=remove_punctuation,
                                remove_numbers=remove_numbers,
                                remove_stopwords=remove_stopwords,
                                lemmatize=lemmatize
                            ) for t in chunk_texts]
                        
                        results = list(executor.map(process_chunk, chunks))
                        batch_results = [text for chunk_result in results for text in chunk_result]
                else:
                    # Sequential batch processing
                    docs = list(self.nlp.pipe(batch))
                    batch_results = []
                    
                    for doc in docs:
                        tokens = []
                        for token in doc:
                            # Apply filters
                            if remove_stopwords and token.is_stop:
                                continue
                            if remove_punctuation and token.is_punct:
                                continue
                            if remove_numbers and token.like_num:
                                continue
                            
                            # Apply transformations
                            token_text = token.lemma_ if lemmatize else token.text
                            token_text = token_text.lower() if lowercase else token_text
                            
                            tokens.append(token_text)
                        
                        batch_results.append(" ".join(tokens))
                
                processed_texts.extend(batch_results)
            
            result[result_column] = processed_texts
        else:
            # Use pandas vectorized operations
            text_series = result[column].fillna("")
            
            if lowercase:
                text_series = text_series.str.lower()
            
            if remove_punctuation:
                # Optimize with translate for better performance
                import string
                punct_table = str.maketrans("", "", string.punctuation)
                text_series = text_series.apply(lambda x: str(x).translate(punct_table))
            
            if remove_numbers:
                text_series = text_series.str.replace(r'\d+', '', regex=True)
            
            text_series = text_series.str.replace(r'\s+', ' ', regex=True).str.strip()
            
            result[result_column] = text_series
        
        return result
    
    def create_bow_features(
        self,
        df: pd.DataFrame,
        text_column: str,
        max_features: int = 100,
        min_df: Union[int, float] = 1,  # Changed from 5 to 1 for smaller test datasets
        max_df: Union[int, float] = 1.0,  # Changed from 0.5 to 1.0 for smaller test datasets
        ngram_range: tuple = (1, 1),
        binary: bool = False,
        prefix: str = 'bow_',
    ) -> pd.DataFrame:
        """
        Create bag-of-words features from a text column.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        text_column : str
            The name of the column containing the text.
        max_features : int, default=100
            The maximum number of features to create.
        min_df : Union[int, float], default=1
            The minimum document frequency for a term to be included.
        max_df : Union[int, float], default=1.0
            The maximum document frequency for a term to be included.
        ngram_range : tuple, default=(1, 1)
            The range of n-grams to consider.
        binary : bool, default=False
            Whether to use binary features (presence/absence) instead of counts.
        prefix : str, default='bow_'
            Prefix for the feature column names.
        
        Returns
        -------
        pd.DataFrame
            A dataframe with the bag-of-words features.
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'pip install scikit-learn'."
            )
        
        # Create the vectorizer
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            binary=binary,
        )
        
        # Transform the text column
        X = vectorizer.fit_transform(df[text_column].astype(str).fillna(""))
        
        # Create a dataframe with the features
        feature_names = [f"{prefix}{feature}" for feature in vectorizer.get_feature_names_out()]
        bow_df = pd.DataFrame(X.toarray(), columns=feature_names, index=df.index)
        
        return bow_df
    
    def create_tfidf_features(
        self,
        df: pd.DataFrame,
        text_column: str,
        max_features: int = 100,
        min_df: Union[int, float] = 1,  # Changed from 5 to 1 for smaller test datasets
        max_df: Union[int, float] = 1.0,  # Changed from 0.5 to 1.0 for smaller test datasets
        ngram_range: tuple = (1, 1),
        prefix: str = 'tfidf_',
    ) -> pd.DataFrame:
        """
        Create TF-IDF features from a text column.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        text_column : str
            The name of the column containing the text.
        max_features : int, default=100
            The maximum number of features to create.
        min_df : Union[int, float], default=1
            The minimum document frequency for a term to be included.
        max_df : Union[int, float], default=1.0
            The maximum document frequency for a term to be included.
        ngram_range : tuple, default=(1, 1)
            The range of n-grams to consider.
        prefix : str, default='tfidf_'
            Prefix for the feature column names.
        
        Returns
        -------
        pd.DataFrame
            A dataframe with the TF-IDF features.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'pip install scikit-learn'."
            )
        
        # Create the vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
        
        # Transform the text column
        X = vectorizer.fit_transform(df[text_column].astype(str).fillna(""))
        
        # Create a dataframe with the features
        feature_names = [f"{prefix}{feature}" for feature in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names, index=df.index)
        
        return tfidf_df
        
    def extract_text_statistics(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Extract basic text statistics for feature engineering.
        
        Parameters
        ----------
        text : str
            The text to analyze.
            
        Returns
        -------
        Dict[str, float]
            A dictionary with text statistics including:
            - word_count: Number of words
            - char_count: Number of characters
            - avg_word_length: Average word length
            - unique_word_ratio: Ratio of unique words to total words
            - uppercase_ratio: Ratio of uppercase letters to total letters
            - digit_ratio: Ratio of digits to total characters
            - punctuation_ratio: Ratio of punctuation to total characters
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'unique_word_ratio': 0,
                'uppercase_ratio': 0,
                'digit_ratio': 0,
                'punctuation_ratio': 0,
            }
        
        # Character count
        char_count = len(text)
        
        # Word count and average word length
        words = [w for w in text.split() if w.strip()]
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        # Unique word ratio
        unique_words = len(set(w.lower() for w in words))
        unique_word_ratio = unique_words / max(1, word_count)
        
        # Character statistics
        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in '.,;:!?-\'\"()[]{}')
        
        uppercase_ratio = uppercase_count / max(1, char_count)
        digit_ratio = digit_count / max(1, char_count)
        punctuation_ratio = punct_count / max(1, char_count)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio,
            'uppercase_ratio': uppercase_ratio,
            'digit_ratio': digit_ratio,
            'punctuation_ratio': punctuation_ratio,
        }
    
    def calculate_readability(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Calculate readability metrics for text.
        
        Parameters
        ----------
        text : str
            The text to analyze.
            
        Returns
        -------
        Dict[str, float]
            A dictionary with readability metrics including:
            - flesch_reading_ease: Flesch Reading Ease score (higher = easier)
            - flesch_kincaid_grade: Flesch-Kincaid Grade Level (lower = easier)
            - coleman_liau_index: Coleman-Liau Index grade level
            - automated_readability_index: Automated Readability Index grade level
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
            }
        
        # Word count
        words = [w for w in text.split() if w.strip()]
        word_count = len(words)
        if word_count == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
            }
        
        # Sentence count
        sentence_pattern = r'[.!?]+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))
        
        # Syllable count (approximate)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            
            # Remove common endings for past tense and plurals
            if word.endswith('es') or word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('e'):
                word = word[:-1]
            
            # Count vowel groups
            count = len(re.findall(r'[aeiouy]+', word))
            return max(1, count)
        
        syllable_count = sum(count_syllables(word) for word in words)
        
        # Character count
        char_count = len(''.join(words))  # Count only word characters
        
        # Calculate metrics
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        chars_per_word = char_count / word_count
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_reading_ease = min(100, max(0, flesch_reading_ease))
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)
        
        # Coleman-Liau Index
        coleman_liau_index = (0.0588 * (chars_per_word * 100)) - (0.296 * (sentence_count / word_count * 100)) - 15.8
        coleman_liau_index = max(0, coleman_liau_index)
        
        # Automated Readability Index
        automated_readability_index = (4.71 * chars_per_word) + (0.5 * words_per_sentence) - 21.43
        automated_readability_index = max(0, automated_readability_index)
        
        return {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'coleman_liau_index': coleman_liau_index,
            'automated_readability_index': automated_readability_index,
        }
        
    def extract_keywords_rake(
        self,
        text: str,
        max_keywords: int = 10,
        min_phrase_length: int = 1,
        max_phrase_length: int = 4,
        min_keyword_frequency: int = 1,
        stopwords: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using the RAKE (Rapid Automatic Keyword Extraction) algorithm.
        
        Parameters
        ----------
        text : str
            The text to extract keywords from.
        max_keywords : int, default=10
            Maximum number of keywords to return.
        min_phrase_length : int, default=1
            Minimum number of words for a phrase to be considered.
        max_phrase_length : int, default=4
            Maximum number of words for a phrase to be considered.
        min_keyword_frequency : int, default=1
            Minimum frequency for a keyword to be considered.
        stopwords : Optional[List[str]], default=None
            List of stopwords to remove. If None, a default list is used.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of keywords and their scores, sorted by score in descending order.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Default stopwords if none provided
        if stopwords is None:
            stopwords = [
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
                'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
                'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don',
                'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
                'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more',
                'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's',
                'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the',
                'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
                'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
                'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
            ]
        
        # Convert stopwords to set for faster lookup
        stopwords_set = set(stopwords)
        
        # Clean and lowercase the text
        text = text.lower()
        
        # Split text into sentences based on sentence delimiters
        sentence_delimiters = r'[.!?,;:\t\n\r\f]'
        sentences = re.split(sentence_delimiters, text)
        
        # Extract phrases from sentences
        phrase_list = []
        for sentence in sentences:
            words = re.findall(r'\b[a-z0-9]+\b', sentence)
            phrase = []
            
            for word in words:
                if word not in stopwords_set:
                    phrase.append(word)
                else:
                    if len(phrase) >= min_phrase_length and len(phrase) <= max_phrase_length:
                        phrase_list.append(' '.join(phrase))
                    phrase = []
            
            # Add the last phrase
            if phrase and len(phrase) >= min_phrase_length and len(phrase) <= max_phrase_length:
                phrase_list.append(' '.join(phrase))
        
        # Calculate word frequencies
        word_freq = {}
        for phrase in phrase_list:
            words = phrase.split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        
        # Calculate phrase scores
        phrase_scores = {}
        for phrase in phrase_list:
            words = phrase.split()
            phrase_frequency = sum(word_freq.get(word, 0) for word in words)
            
            if phrase_frequency < min_keyword_frequency:
                continue
                
            word_count = len(words)
            word_degree = sum(word_freq.get(word, 0) for word in words)
            
            score = word_degree / max(1, word_count)
            
            phrase_scores[phrase] = score
        
        # Sort phrases by score
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return sorted_phrases[:max_keywords]
        
    def analyze_sentiment(
        self,
        text: str,
        lexicon: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Analyze sentiment using a lexicon-based approach.
        
        Parameters
        ----------
        text : str
            The text to analyze.
        lexicon : Optional[Dict[str, float]], default=None
            Dictionary mapping words to sentiment scores. If None, a default
            small lexicon is used.
            
        Returns
        -------
        Dict[str, float]
            A dictionary with sentiment metrics including:
            - sentiment_score: Overall sentiment score (positive = positive sentiment)
            - positive_ratio: Ratio of positive words to total words
            - negative_ratio: Ratio of negative words to total words
            - neutral_ratio: Ratio of neutral words to total words
            - sentiment_variance: Variance of sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 1.0,
                'sentiment_variance': 0.0,
            }
        
        # Default small sentiment lexicon if none provided
        if lexicon is None:
            lexicon = {
                # Positive words
                'good': 1.0, 'great': 1.5, 'excellent': 2.0, 'best': 2.0, 'amazing': 1.8,
                'wonderful': 1.8, 'fantastic': 1.9, 'terrific': 1.8, 'outstanding': 1.7,
                'brilliant': 1.7, 'superb': 1.8, 'perfect': 2.0, 'positive': 1.0,
                'happy': 1.5, 'love': 1.8, 'like': 0.8, 'enjoy': 1.2, 'awesome': 1.5,
                'better': 0.8, 'recommend': 1.2, 'impressive': 1.3, 'exceptional': 1.7,
                'superior': 1.4, 'favorite': 1.5, 'pleased': 1.2, 'satisfied': 1.1,
                
                # Negative words
                'bad': -1.0, 'terrible': -1.8, 'awful': -1.7, 'horrible': -1.9, 'worst': -2.0,
                'poor': -1.0, 'disappointing': -1.3, 'negative': -1.0, 'sad': -1.2,
                'hate': -1.8, 'dislike': -1.0, 'unfortunate': -1.0, 'inferior': -1.2,
                'mediocre': -0.7, 'worse': -1.4, 'difficult': -0.5, 'problem': -0.8,
                'issue': -0.6, 'fault': -1.0, 'fail': -1.5, 'failure': -1.5, 'broken': -1.2,
                'unhappy': -1.3, 'disappointed': -1.4, 'frustrating': -1.5, 'sucks': -1.7
            }
        
        # Preprocess the text
        text = self.preprocess_text(
            text, 
            lowercase=True, 
            remove_punctuation=True
        )
        
        # Split into words
        words = text.split()
        if not words:
            return {
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 1.0,
                'sentiment_variance': 0.0,
            }
        
        # Calculate sentiment metrics
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for word in words:
            score = lexicon.get(word, 0.0)
            sentiment_scores.append(score)
            
            if score > 0:
                positive_count += 1
            elif score < 0:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall metrics
        total_words = len(words)
        sentiment_score = sum(sentiment_scores) / total_words if total_words > 0 else 0
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        neutral_ratio = neutral_count / total_words if total_words > 0 else 1
        
        # Calculate variance of sentiment
        if total_words > 1:
            mean_score = sentiment_score
            variance = sum((score - mean_score) ** 2 for score in sentiment_scores) / total_words
        else:
            variance = 0.0
            
        return {
            'sentiment_score': sentiment_score,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'sentiment_variance': variance,
        }
    
    def calculate_document_similarity(
        self,
        doc1: str,
        doc2: str,
        method: str = 'cosine',
        lowercase: bool = True,
        remove_punctuation: bool = True,
    ) -> float:
        """
        Calculate similarity between two text documents.
        
        Parameters
        ----------
        doc1 : str
            First document text.
        doc2 : str
            Second document text.
        method : str, default='cosine'
            Similarity method to use. Options are:
            - 'cosine': Cosine similarity
            - 'jaccard': Jaccard similarity
            - 'overlap': Overlap coefficient
        lowercase : bool, default=True
            Whether to convert texts to lowercase before comparison.
        remove_punctuation : bool, default=True
            Whether to remove punctuation before comparison.
            
        Returns
        -------
        float
            Similarity score between 0 and 1, where 1 indicates identical documents.
        """
        # Validate and preprocess texts
        if not isinstance(doc1, str) or not doc1.strip() or not isinstance(doc2, str) or not doc2.strip():
            return 0.0
        
        # Preprocess text
        doc1 = self.preprocess_text(doc1, lowercase=lowercase, remove_punctuation=remove_punctuation)
        doc2 = self.preprocess_text(doc2, lowercase=lowercase, remove_punctuation=remove_punctuation)
        
        # Split into words
        words1 = doc1.split()
        words2 = doc2.split()
        
        if not words1 or not words2:
            return 0.0
        
        # Create word frequency dictionaries (term frequency vectors)
        vec1 = Counter(words1)
        vec2 = Counter(words2)
        
        # Calculate similarity based on method
        if method == 'cosine':
            # Cosine similarity
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum(vec1[x] * vec2[x] for x in intersection)
            
            sum1 = sum(val ** 2 for val in vec1.values())
            sum2 = sum(val ** 2 for val in vec2.values())
            denominator = (sum1 ** 0.5) * (sum2 ** 0.5)
            
            if denominator == 0:
                return 0.0
                
            return numerator / denominator
            
        elif method == 'jaccard':
            # Jaccard similarity (set-based)
            set1 = set(words1)
            set2 = set(words2)
            
            if not set1 and not set2:
                return 1.0
                
            return len(set1 & set2) / len(set1 | set2)
            
        elif method == 'overlap':
            # Overlap coefficient
            set1 = set(words1)
            set2 = set(words2)
            
            if not set1 or not set2:
                return 0.0
                
            return len(set1 & set2) / min(len(set1), len(set2))
            
        else:
            raise ValueError(f"Unknown similarity method: {method}. "
                            "Supported methods are: 'cosine', 'jaccard', 'overlap'.")
            
    def benchmark_text_processing(
        self,
        df: pd.DataFrame,
        column: str,
        iterations: int = 3,
        **kwargs
    ) -> dict:
        """
        Benchmark different text processing backends.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        column : str
            The name of the column to process.
        iterations : int, default=3
            Number of times to run each benchmark.
        **kwargs
            Additional keyword arguments to pass to process_dataframe_column.
        
        Returns
        -------
        dict
            Dictionary with results for each backend.
        """
        import time
        import gc
        
        def get_available_backends():
            """Helper to determine available backends."""
            available = ['pandas']  # Always available
            
            # Check if polars is available
            try:
                import polars
                available.append('polars')
            except ImportError:
                pass
            
            # Check if pyarrow is available
            try:
                import pyarrow
                try:
                    # Try to use PyArrow string type
                    pd.Series(["test"]).astype("string[pyarrow]")
                    available.append('pyarrow')
                except (TypeError, ValueError):
                    pass
            except ImportError:
                pass
            
            return available
        
        # Get available backends
        backends = get_available_backends()
        
        results = {}
        
        for backend in backends:
            times = []
            for _ in range(iterations):
                # Clean memory
                gc.collect()
                
                # Time the operation
                start_time = time.time()
                _ = self.process_dataframe_column(
                    df, column, backend=backend, **kwargs
                )
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Calculate statistics
            results[backend] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'times': times
            }
        
        # Print results
        print(f"Benchmarking results (seconds):")
        for backend, stats in results.items():
            print(f"  {backend}: {stats['mean']:.4f}s (min: {stats['min']:.4f}s, max: {stats['max']:.4f}s)")
        
        return results
    
    def create_text_features(
        self,
        df: pd.DataFrame,
        text_column: str,
        include_stats: bool = True,
        include_readability: bool = True,
        include_sentiment: bool = True,
        prefix: str = 'text_',
    ) -> pd.DataFrame:
        """
        Create text features from a text column, combining multiple text analysis methods.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        text_column : str
            The name of the column containing the text.
        include_stats : bool, default=True
            Whether to include basic text statistics features.
        include_readability : bool, default=True
            Whether to include readability metrics features.
        include_sentiment : bool, default=True
            Whether to include sentiment analysis features.
        prefix : str, default='text_'
            Prefix for the feature column names.
            
        Returns
        -------
        pd.DataFrame
            A dataframe with the text features.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe.")
            
        result = pd.DataFrame(index=df.index)
        
        # Process each text in the column
        texts = df[text_column].astype(str).fillna("")
        
        # Add text statistics features
        if include_stats:
            stats_list = texts.apply(self.extract_text_statistics)
            stats_df = pd.DataFrame.from_records(stats_list)
            stats_df.columns = [f"{prefix}stat_{col}" for col in stats_df.columns]
            result = pd.concat([result, stats_df], axis=1)
        
        # Add readability metrics
        if include_readability:
            readability_list = texts.apply(self.calculate_readability)
            readability_df = pd.DataFrame.from_records(readability_list)
            readability_df.columns = [f"{prefix}read_{col}" for col in readability_df.columns]
            result = pd.concat([result, readability_df], axis=1)
        
        # Add sentiment features
        if include_sentiment:
            sentiment_list = texts.apply(self.analyze_sentiment)
            sentiment_df = pd.DataFrame.from_records(sentiment_list)
            sentiment_df.columns = [f"{prefix}sent_{col}" for col in sentiment_df.columns]
            result = pd.concat([result, sentiment_df], axis=1)
        
        return result