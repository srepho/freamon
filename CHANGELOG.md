# Changelog

## Version 0.3.19 - EDA Reporting Enhancements
- Fixed accordion functionality issues in HTML reports with improved JavaScript
- Optimized image sizes in reports to significantly reduce file size
- Refactored report generation code for better maintainability
- Improved HTML structure for better Bootstrap 5 compatibility
- Added proper error handling for analysis section failures
- Enhanced ID sanitization for better HTML/JavaScript compatibility
- Added detailed debugging output for easier troubleshooting

## Version 0.3.18 - Text Deduplication Module
- Added comprehensive text deduplication functionality with dedicated modules
- Implemented exact deduplication methods: hash-based and n-gram fingerprinting
- Added fuzzy deduplication with Levenshtein, Jaccard, cosine, and embedding similarity
- Implemented scalable deduplication with LSH (Locality-Sensitive Hashing)
- Added document fingerprinting with MinHash and SimHash
- Added clustering-based deduplication with hierarchical and DBSCAN methods
- Created example script with benchmarking for different deduplication methods
- Integrated with existing TextProcessor functionality for seamless use

## Version 0.3.17 - Enhanced EDA for Large Datasets & Jupyter Export
- Added smart data table rendering showing first/last rows with row count indicators
- Implemented lazy loading for images to improve HTML report performance
- Added client-side export to Jupyter notebook functionality
- Enhanced EDAAnalyzer.generate_report with new parameters:
  - lazy_loading: Enable/disable lazy loading for images
  - include_export_button: Add a button to export report as Jupyter notebook
- Updated return type of generate_report to include the HTML string
- Added comprehensive documentation for large dataset handling
- Created new example scripts:
  - large_dataset_eda_example.py: Demonstrates all optimizations
  - jupyter_export_example.py: Shows Jupyter notebook export feature
  - eda_performance_test.py: Benchmarks report generation performance

## Version 0.3.16 - EDA Reporting Robustness Fix
- Fixed crashes in EDA reporting when data contains currency symbols
- Enhanced matplotlib patches to handle problematic characters like $, _, ^, etc.
- Added robust error handling for mathtext parsing failures
- Implemented fallback mechanisms for EDA analysis steps
- Created minimal report generation for severe error cases
- Made accordion components more robust in HTML reports
- Added comprehensive text preprocessing for safe HTML report generation
- Created utility function to easily apply patches (patch_freamon)
- Added example script demonstrating usage with financial data containing currency values

## Version 0.3.15 - Text Processor Train/Test Split Support
- Added fit/transform capability to TextProcessor for TF-IDF and bag-of-words features
- Implemented vectorizer storage in TextProcessor instances
- Added transform_tfidf_features and transform_bow_features methods for test data
- Improved text preprocessing workflow for train/test splits
- Updated documentation and examples for text feature extraction

## Version 0.3.14 - Display and HTML Report Improvements
- Fixed matplotlib LaTeX parsing errors when displaying text with dollar signs
- Added utility functions to safely handle currency symbols in visualizations
- Fixed HTML report accordion components not expanding/collapsing correctly in Bootstrap 5
- Added proper JavaScript initialization for HTML accordions with event handlers
- Improved CSS styling for accordion components in reports
- Changed default for multivariate analysis from True to False to improve performance
- Updated documentation to reflect new defaults and performance considerations

## Version 0.3.13 - Feature Importance Analysis
- Added feature importance calculation for target variables using machine learning
- Implemented support for both classification and regression targets
- Added multiple importance methods: random forest, permutation, and SHAP
- Created new Feature Importance tab in EDA reports
- Enhanced EDA analyzer with feature importance analysis
- Added comprehensive test coverage for the new functionality

## Version 0.3.12 - Performance Optimizations
- Added data sampling option for large datasets in EDA analysis
- Implemented caching for expensive operations like PCA and t-SNE
- Added progress monitoring with execution time tracking
- Enhanced HTML reports to display sampling information
- Added support for parallel processing and improved multivariate analysis performance

## Version 0.3.11 - Matplotlib Warning Fixes
- Fixed matplotlib warnings about using categorical units for numeric data
- Resolved FutureWarnings related to pandas groupby observed parameter
- Enhanced year, month, and weekday visualization to avoid matplotlib warnings
- Improved chart display for datetime distributions

## Version 0.3.10 - EDA Datetime Plotting Fix
- Fixed error in EDA datetime plotting with floating-point year values (like '2024.0')
- Enhanced robustness of year value handling in EDA charts
- Improved error handling in datetime visualization
- Added graceful fallback for unparseable year values

## Version 0.3.9 - Enhanced Month-Year Format Detection
- Fixed month-year format detection for columns with missing values
- Improved detection algorithm to use adaptive thresholds based on missing data percentages
- Enhanced conversion of month-year formats with better handling of string types
- Fixed conversion errors with numpy.str_ types in date formats 
- Reduced minimum required valid values for month-year detection to 3 (from 5)
- Updated test assertions to accommodate the improved detection behavior

## Version 0.3.8 - Month-Year Format Detection Fix
- Fixed issue with month-year format detection for formats like 'Jun-24', 'Jul-24'
- Added explicit month-year format check for datetime columns to ensure correct semantic type is applied
- Enhanced month-year format detection to work with columns containing missing values
- Added debug logging for better diagnostics

## Version 0.3.7 - PyArrow Integration and Performance Optimization
- Added PyArrow integration for faster dataframe operations
- Optimized type detection for large dataframes with improved sampling
- Enhanced categorical detection with cardinality threshold
- Fixed memory usage issues with large dataframes

## Version 0.3.6 - Enhanced Datatype Detection
- Added support for mixed date formats within a single column
- Improved handling of datetime conversions with missing values
- Enhanced detection of numeric patterns in text columns
- Updated conversion suggestions for better type optimization

## Version 0.3.5 - Scientific Notation Support
- Added detection for scientific notation in numeric columns
- Enhanced reporting for numeric patterns
- Improved performance with optimized sampling

## Version 0.3.4 - Date Format Enhancements
- Added detection for mixed date formats
- Added support for scientific notation in numeric values
- Improved performance for large dataframes

## Version 0.3.3 - Excel Date Detection
- Added detection for Excel date numbers
- Enhanced reporting for datetime conversions
- Fixed timezone handling in datetime conversion

## Version 0.3.2 - Australian Pattern Support
- Added detection for Australian postal codes
- Added Australian phone number patterns
- Added support for Australian business identifiers (ABN, ACN, TFN)

## Version 0.3.1 - Performance Improvements
- Optimized detection for large dataframes
- Reduced memory usage with improved sampling
- Added parallel processing for multi-column dataframes

## Version 0.3.0 - Initial Release
- Basic type detection (numeric, string, datetime)
- Semantic type detection (email, URL, phone, etc.)
- Conversion suggestions for optimal storage types
- Support for categorical data identification