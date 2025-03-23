# Changelog

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