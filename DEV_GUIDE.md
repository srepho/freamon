# Freamon Package Planning Document

**Package Name (Working Title):** freamon

**Target Python Versions:** 3.10, 3.11, 3.12

## 1. Mission Statement and Scope

*   **Mission:** To provide a comprehensive, user-friendly, and efficient Python package for end-to-end machine learning on tabular data, simplifying complex data science tasks from initial exploration to model building, with a focus on both standard tabular and time-series datasets.
*   **Scope:**
    *   Data Quality Assessment and Reporting (HTML)
    *   Exploratory Data Analysis (EDA) and Visualization (HTML)
    *   Feature Engineering (automated and manual)
    *   Feature Importance Analysis
    *   Model Training (with extensibility)
    *   Model Evaluation and Validation
    *   Train/Test Splitting (including time-series aware splitting)
    *   Handling of datasets larger than RAM (out-of-core processing)
    *   Seamless integration with popular data manipulation libraries.
*   **Out of Scope (for initial release):**
    *   Deep Learning
    *   Natural Language Processing (NLP)
    *   Computer Vision
    *   Deployment to production environments
    *   Support for database connections, data streaming, or real-time predictions.

## 2. Core Modules and Functionality

*   **`data_quality`:**
    *   Generates an HTML report covering:
        *   **Missing Value Analysis:** Counts, percentages, visualizations. Handles various missing value representations.
        *   **Data Type Checks:** Identifies inconsistent data types. Suggests conversions.
        *   **Outlier Detection:** Provides options for various outlier detection methods with configurable thresholds.
        *   **Duplicate Detection:** Identifies and optionally removes duplicate rows.
        *   **Cardinality Analysis:** Reports on unique value counts.
        *   **Descriptive Statistics:** Summary statistics for numerical and categorical features.
    *   Internal functions for outlier detection (e.g., IQR, Z-score).
    *   Function to handle missing data ('drop', 'mean', 'median', 'mode', 'constant', 'interpolate').
    *   Function to validate dataframe structure.

*   **`eda`:**
    *   Generates an HTML report with:
        *   **Univariate Analysis:** Distribution plots for numerical features. Value counts for categorical features.  Static and interactive plots.
        *   **Bivariate Analysis:** Visualizations to explore relationships between features and the target (if provided). Handles different feature type combinations.
        *   **Time Series Analysis (if time index is present):** Line plots, autocorrelation plots, seasonal decomposition.
    *   Internal functions for generating various plots.
    *   Internal function for autocorrelation calculation.

*   **`features`:**
    *   A class to manage feature engineering, supporting method chaining.
        *   Create interaction terms.
        *   Encode categorical features (one-hot, label, target encoding with smoothing).
        *   Bin numerical features.
        *   Extract time-based features from datetime columns.
        *   Apply scaling methods.
        *   Transform a dataframe by applying accumulated steps.
        *   Fit scalers and encoders.
    *   Calculates and visualizes feature importance.
    *   Added functionality to create polynomial features.

*   **`model_selection`:**
    *   Splits data into training and testing sets.
        *   Time-series aware splitting option, handling uneven time series lengths, using a grouping column.
        *   Stratified splitting option.
        *   Option to create time series folds
    *   Performs time-series cross-validation, respecting temporal order and handling multiple time series.
    *   Internal function to validate the time series split column.

*   **`modeling`:**
    *   A class for model training.
        *   Trains specified models. Supports hyperparameter passing.
        *   Generates predictions.
        *   Evaluates the model with various metrics (classification and regression).
        *   Returns the underlying trained model object.
    *   Internal function to instantiate the correct model.

*   **`utils`:**
    *   Functions to convert between different dataframe types.
    *   Functions to check the dataframe type.
    *   Function to check data size and suggest settings
    *   Function to optimize column data types.
    *   Function to estimate dataframe memory useage.
    *   Validation function.

## 3. Data Handling Strategy (Out-of-Core)

*   **Prioritize a library with lazy evaluation:** Utilize a library known for its lazy evaluation and out-of-core capabilities.
*   **Chunking:** Implement chunking for operations requiring data in memory.
*   **Data Type Optimization:** Automatically downcast numerical data types.
*   **Sparse Matrices:** Consider sparse representations for data with many zeros.
*   **Memory profiling:** Add checks to monitor memory use

## 4. API Design Principles

*   **Consistency:** Consistent naming, parameters, and return types.
*   **Flexibility:** Customizable behavior through parameters.
*   **User-Friendliness:** Clear error messages, docstrings, and examples.
*   **Fluent Interface (Method Chaining):** Enable where appropriate.
*   **Type Hinting:** Use throughout for maintainability.
*   **Dataframe Agnostic:** Functions should accept multiple dataframe types.

## 5. Testing Strategy

*   **Unit Tests:** Comprehensive unit tests for all functions and classes. Test edge cases and both supported dataframe types.
*   **Integration Tests:** Test interactions between modules.
*   **Doctests:** Include in docstrings for examples and documentation updates.
*   **Performance Benchmarking:** Measure performance of key functions.
*   **Test Data:** Create synthetic datasets with various characteristics.
*   **CI/CD:** Integrate with a CI/CD system.

## 6. Documentation

*   Use a documentation generator to produce:
    *   API Reference (from docstrings).
    *   User Guide: Tutorials and examples.
    *   Conceptual Overview.
    *   Contribution Guidelines.
*   Host documentation online.
*   Provide numerous, clear examples.
*   Comprehensive docstrings for all public elements.

## 7. Development Workflow

1.  **Feature Branching:** Use Git, with new branches for features/fixes.
2.  **Pull Requests:** Submit for code review before merging.
3.  **Code Reviews:** Thoroughly review all code changes.
4.  **Continuous Integration:** Automate testing and linting.
5.  **Versioning:** Use semantic versioning.

## 8. Future Enhancements (Beyond Initial Release)

*   Deep Learning Integration.
*   Model Explainability.
*   Automated Model Selection (AutoML).
*   Data Drift Detection.
*   Deployment Helpers.
*   Interactive Dashboards.
*   Support for other data formats

## 9. Project Timeline (Example)

*   **Phase 1 (1-2 months):** Project setup, `data_quality` module, initial documentation.
*   **Phase 2 (2-3 months):** `eda` module, basic `features` module, documentation updates.
*   **Phase 3 (2-3 months):** `model_selection` and `modeling`, advanced features, time-series splitting.
*   **Phase 4 (1-2 months):** Out-of-core optimization, testing, documentation, initial release.

## 10. Detailed Module Implementation (Continued)

*   **`utils` (Continued)**
    *   Infer and suggest optimal data types.
    *   Validate input dataframe types.
    *   Estimate dataframe memory usage.

*   **`data_quality` (Continued)**
    *   **Missing Value Handling:** Handle different missing value representations. Allow custom representations.
    *   **HTML Report:** Flexible, customizable templates. Interactive elements. Customizable appearance.

*   **`eda` (Continued)**
    *   **Time Series Plots:** ACF, PACF, seasonal decomposition. Handle irregular time series.
    *   **Interactive Visualizations:** Prioritize interactive plots.  Consider interactive exploration tools.

*   **`features` (Continued)**
    *   **Feature Scaling:** Store scaler parameters for consistent application.  Consider column-specific transformations.
    *   **Categorical Encoding:** Target encoding with smoothing to prevent overfitting.
    *   **Text Features (Future):** Basic text feature engineering.

*   **`model_selection` (Continued)**
    *   **Time Series Splitting:** Handle uneven series lengths. Group by an identifier, split each group, combine. Consider standard time series splitters for cross-validation.
    *   Add a gap option for time series cross validation
    *   **Stratified Splitting:** Use appropriate methods for stratification.

*   **`modeling` (Continued)**
    *   **Model Persistence:** Save and load trained models.
    *   **Extensibility:** Design for easy addition of new models.
    *   **Hyperparameter Optimization (Future).**

## 11. Handling Out-of-Core Processing (Detailed)

*   **Chunking:** Determine chunk size, iterate in chunks, apply operations, aggregate results. User-configurable chunk size.
*   **Integration with a distributed computing library (Future).**

## 12. Error Handling and Validation

*   **Informative Exceptions:** Clear error messages.
*   **Input Validation:** Validate dataframe types and preconditions.
*   **Type Checking:** Use type hints and a static type checker.

## 13. Performance Optimization

*   **Profiling:** Identify performance bottlenecks.
*   **Vectorized Operations:** Avoid explicit loops.
*   **Data Type Optimization:** Downcast numerical types.
*   **Lazy Evaluation:** Minimize unnecessary computations.
*   **Parallel Processing:** Parallelize tasks.

## 14. Advanced Features (Future Considerations)

*   Automated Feature Selection.
*   Anomaly Detection.
*   Pipeline Building.
*   Interactive Data Exploration.

## 15. Addressing Potential Challenges

*   **Maintaining Compatibility:** Keep up with library updates. Comprehensive testing.
*   **Memory Management:** Manage memory, provide guidance.
*   **Performance Tuning:** Monitor and optimize.
*   **User Adoption:** Clear documentation, examples, tutorials.