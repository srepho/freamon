"""
Utility functions for text processing.
"""
from typing import Any, Dict, List, Optional, Union

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
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a text column in a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the text column.
        column : str
            The name of the column to process.
        result_column : Optional[str], default=None
            The name of the column to store the processed text.
            If None, the original column is overwritten.
        **kwargs
            Additional keyword arguments to pass to preprocess_text.
        
        Returns
        -------
        pd.DataFrame
            The dataframe with the processed text column.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        
        result = df.copy()
        
        # Process each text in the column
        processed_texts = result[column].astype(str).apply(
            lambda x: self.preprocess_text(x, **kwargs)
        )
        
        # Determine the output column
        if result_column is None:
            result_column = column
        
        # Store the processed texts
        result[result_column] = processed_texts
        
        return result
    
    def create_bow_features(
        self,
        df: pd.DataFrame,
        text_column: str,
        max_features: int = 100,
        min_df: Union[int, float] = 5,
        max_df: Union[int, float] = 0.5,
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
        min_df : Union[int, float], default=5
            The minimum document frequency for a term to be included.
        max_df : Union[int, float], default=0.5
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
        min_df: Union[int, float] = 5,
        max_df: Union[int, float] = 0.5,
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
        min_df : Union[int, float], default=5
            The minimum document frequency for a term to be included.
        max_df : Union[int, float], default=0.5
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