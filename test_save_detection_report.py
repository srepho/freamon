import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

# Create a test DataFrame with various data types
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
    'date_joined': ['2023-01-15', '2022-11-30', '2023-03-22', '2022-08-17', '2023-02-05'],
    'balance': ['$1,234.56', '$987.65', '$2,345.67', '$456.78', '$3,456.78'],
    'active': [True, False, True, True, False],
    'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
              'alice@example.com', 'charlie@example.com'],
    'age': [35, 42, 28, 39, 45],
    'score': [87.5, 92.3, 76.8, 95.1, 81.9],
    'last_login': ['2023-05-10 14:30:25', '2023-05-09 09:15:40', '2023-05-11 16:45:12',
                  '2023-05-08 11:20:35', '2023-05-12 13:10:50'],
    'mixed_data': ['abc123', '2023-01-01', '$123.45', '75%', 'regular text']
})

# Create the DataTypeDetector and detect all types
print("Creating DataTypeDetector and detecting types...")
detector = DataTypeDetector(df)
detector.detect_all_types()

# Test the new save_html_report method
print("Testing the new save_html_report method...")
report_path = detector.save_html_report("datatype_detection_report.html", include_stats=True)
print(f"Report saved to: {report_path}")

# Test the new get_column_report_html method
print("\nTesting the new get_column_report_html method...")
html_content = detector.get_column_report_html()
print(f"Generated HTML content length: {len(html_content)} characters")

# Save the HTML content from get_column_report_html to a file
custom_report_path = "custom_datatype_report.html"
with open(custom_report_path, "w") as f:
    f.write(html_content)
print(f"Custom report saved to: {custom_report_path}")

# Display a summary of the detected types
print("\nDetected column types:")
for col, type_info in detector.column_types.items():
    print(f"{col}: {type_info}")

print("\nSemantic types:")
for col, sem_type in detector.semantic_types.items():
    if sem_type:
        print(f"{col}: {sem_type}")

print("\nSuggested conversions:")
for col, suggestion in detector.conversion_suggestions.items():
    print(f"{col}: {suggestion}")