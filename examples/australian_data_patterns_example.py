"""
Australian Data Patterns Detection Example

This example demonstrates Freamon's ability to detect and handle Australian-specific data patterns,
including:

1. Australian postcodes (4 digits, with proper zero-padding)
2. Australian phone numbers (landline and mobile)
3. Australian Business Numbers (ABNs)
4. Australian Company Numbers (ACNs)
5. Australian Tax File Numbers (TFNs)

The example shows how these patterns are detected, validated, and converted to appropriate formats.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types

# Create a sample dataset with Australian data patterns
def create_australian_sample_data(rows=100):
    """Create a sample dataset with Australian data formats."""
    
    # Australian postcodes (4 digits, 0-9999)
    # Examples of real ranges:
    # NSW: 1000-2999, ACT: 0200-0299 and 2600-2639, VIC: 3000-3999
    # QLD: 4000-4999, SA: 5000-5999, WA: 6000-6999, TAS: 7000-7999
    # NT: 0800-0899
    
    # Create both string and integer postcodes (to simulate data issues)
    nsw_postcodes = [f"{np.random.randint(1000, 2999)}" for _ in range(rows//5)]
    vic_postcodes = [f"{np.random.randint(3000, 3999)}" for _ in range(rows//5)]
    qld_postcodes = [f"{np.random.randint(4000, 4999)}" for _ in range(rows//5)]
    nt_postcodes = [f"{np.random.randint(800, 899)}" for _ in range(rows//10)]  # Missing leading zeros
    act_postcodes = [f"{np.random.randint(2600, 2639)}" for _ in range(rows//10)]
    
    # Create integer postcodes that will lose leading zeros
    int_postcodes = []
    for p in nsw_postcodes + vic_postcodes + qld_postcodes + nt_postcodes + act_postcodes:
        int_postcodes.append(int(p))
    
    # Ensure we have the right number of rows by sampling
    str_postcodes = (nsw_postcodes + vic_postcodes + qld_postcodes + nt_postcodes + act_postcodes)[:rows]
    int_postcodes = int_postcodes[:rows]
    
    # Australian phone numbers
    # Landlines: +61 2/3/7/8 XXXX XXXX or 0X XXXX XXXX
    # Mobiles: +61 4XX XXX XXX or 04XX XXX XXX
    area_codes = ['2', '3', '7', '8']  # Sydney, Melbourne, etc.
    landlines = []
    mobiles = []
    
    for _ in range(rows):
        # Create landlines with different formats
        area = np.random.choice(area_codes)
        number = f"{np.random.randint(1000, 9999)} {np.random.randint(1000, 9999)}"
        
        # Randomly choose format
        format_choice = np.random.randint(0, 3)
        if format_choice == 0:
            landlines.append(f"+61 {area} {number}")
        elif format_choice == 1:
            landlines.append(f"0{area} {number}")
        else:
            landlines.append(f"({area}) {number}")
        
        # Create mobiles with different formats
        mobile_prefix = np.random.randint(10, 99)
        mobile_number = f"{np.random.randint(100, 999)} {np.random.randint(100, 999)}"
        
        # Randomly choose format
        format_choice = np.random.randint(0, 2)
        if format_choice == 0:
            mobiles.append(f"+61 4{mobile_prefix} {mobile_number}")
        else:
            mobiles.append(f"04{mobile_prefix} {mobile_number}")
    
    # Australian Business Numbers (ABNs) - 11 digits
    abns = []
    for _ in range(rows):
        # Generate random 11-digit number with spaces
        abn_parts = [str(np.random.randint(10, 99)), 
                     str(np.random.randint(100, 999)), 
                     str(np.random.randint(100, 999)), 
                     str(np.random.randint(100, 999))]
        abns.append(" ".join(abn_parts))
    
    # Australian Company Numbers (ACNs) - 9 digits
    acns = []
    for _ in range(rows):
        # Generate random 9-digit number with spaces
        acn_parts = [str(np.random.randint(100, 999)), 
                     str(np.random.randint(100, 999)), 
                     str(np.random.randint(100, 999))]
        acns.append(" ".join(acn_parts))
    
    # Australian Tax File Numbers (TFNs) - 8 or 9 digits
    tfns = []
    for _ in range(rows):
        # Generate random 8 or 9-digit number with spaces
        if np.random.random() < 0.5:  # 8 digits
            tfn_parts = [str(np.random.randint(100, 999)), 
                         str(np.random.randint(100, 999)), 
                         str(np.random.randint(10, 99))]
        else:  # 9 digits
            tfn_parts = [str(np.random.randint(100, 999)), 
                         str(np.random.randint(100, 999)), 
                         str(np.random.randint(100, 999))]
        tfns.append(" ".join(tfn_parts))
    
    # Create the final dataframe
    df = pd.DataFrame({
        'id': range(1, rows + 1),
        'name': [f"Customer {i}" for i in range(rows)],
        'email': [f"customer{i}@example.com.au" for i in range(rows)],
        'postcode_str': str_postcodes,
        'postcode_int': int_postcodes,
        'phone': landlines,
        'mobile': mobiles,
        'abn': abns,
        'acn': acns,
        'tfn': tfns,
        # Add some other columns
        'amount': np.random.lognormal(6, 1, rows),
        'transaction_count': np.random.randint(1, 50, rows),
        'status': np.random.choice(['Active', 'Inactive', 'Pending'], rows),
    })
    
    # Add missing values to simulate real data
    for col in df.columns:
        if col != 'id':
            mask = np.random.random(rows) < 0.05
            df.loc[mask, col] = np.nan
    
    return df

# Create the sample dataset
print("Creating sample dataset with Australian data patterns...")
df = create_australian_sample_data(200)
print("\nSample data:")
print(df.head())

# Detect data types
print("\nDetecting data types...")
detector = DataTypeDetector(df)
results = detector.detect_all_types()

# Group detection results by semantic type
semantic_types = {}
for col, info in results.items():
    if 'semantic_type' in info:
        sem_type = info['semantic_type']
        if sem_type not in semantic_types:
            semantic_types[sem_type] = []
        semantic_types[sem_type].append(col)

# Print detection results for Australian patterns
print("\nDetected Australian patterns:")
au_patterns = ['au_postcode', 'au_phone', 'au_mobile', 'au_abn', 'au_acn', 'au_tfn']
for pattern in au_patterns:
    if pattern in semantic_types:
        print(f"  {pattern}: {', '.join(semantic_types[pattern])}")

# Print conversion suggestions
print("\nConversion suggestions:")
for col, info in results.items():
    if 'suggested_conversion' in info:
        print(f"  {col}: Convert to {info['suggested_conversion']['convert_to']} using {info['suggested_conversion']['method']}")

# Apply suggested conversions
print("\nApplying conversions...")
converted_df = detector.convert_types()

# Display results before and after conversion for postcodes
print("\nPostcode Comparison (Before vs After):")
postcode_cols = [col for col in df.columns if 'postcode' in col.lower()]
comparison = pd.DataFrame({
    'Original': df[postcode_cols].head(10).values,
    'Converted': converted_df[postcode_cols].head(10).values
})
print(comparison)

# Calculate distribution of postcodes by state
print("\nAnalyzing postcode distribution by state...")
# Function to categorize postcodes by state
def categorize_postcode(postcode):
    if pd.isna(postcode):
        return 'Unknown'
    
    # Convert to string and ensure 4 digits
    try:
        pc_str = str(postcode).strip().zfill(4)
    except:
        return 'Invalid'
    
    # Categorize based on first digits
    prefix = pc_str[:1]
    if pc_str.startswith('0'):
        if pc_str.startswith('08'):
            return 'NT'
        elif pc_str.startswith('02'):
            return 'ACT'
        return 'Other'
    elif pc_str.startswith('1') or pc_str.startswith('2'):
        if pc_str.startswith('26') and int(pc_str) <= 2639:
            return 'ACT'
        return 'NSW'
    elif pc_str.startswith('3'):
        return 'VIC'
    elif pc_str.startswith('4'):
        return 'QLD'
    elif pc_str.startswith('5'):
        return 'SA'
    elif pc_str.startswith('6'):
        return 'WA'
    elif pc_str.startswith('7'):
        return 'TAS'
    else:
        return 'Invalid'

# Apply the categorization and plot distribution
postcode_col = [col for col in converted_df.columns if 'postcode' in col.lower()][0]
converted_df['state'] = converted_df[postcode_col].apply(categorize_postcode)
state_counts = converted_df['state'].value_counts()

# Create a bar chart of state distribution
plt.figure(figsize=(10, 6))
state_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Postcodes by State/Territory')
plt.xlabel('State/Territory')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('au_postcode_distribution.png')
print("Saved postcode distribution plot to 'au_postcode_distribution.png'")

# Analyze phone number types
print("\nAnalyzing phone number types...")
# Function to categorize phone numbers
def categorize_phone(phone):
    if pd.isna(phone):
        return 'Unknown'
    
    phone_str = str(phone).strip().replace(' ', '').replace('-', '')
    
    if phone_str.startswith('+614') or phone_str.startswith('04'):
        return 'Mobile'
    elif (phone_str.startswith('+612') or phone_str.startswith('02') or
          phone_str.startswith('+613') or phone_str.startswith('03') or
          phone_str.startswith('+617') or phone_str.startswith('07') or
          phone_str.startswith('+618') or phone_str.startswith('08')):
        return 'Landline'
    else:
        return 'Other'

# Apply phone categorization
phone_cols = [col for col in converted_df.columns if 'phone' in col.lower() or 'mobile' in col.lower()]
for col in phone_cols:
    converted_df[f'{col}_type'] = converted_df[col].apply(categorize_phone)
    phone_type_counts = converted_df[f'{col}_type'].value_counts()
    print(f"\n{col} types:")
    print(phone_type_counts)

# Create a pie chart of phone types
plt.figure(figsize=(10, 6))
phone_type_combined = pd.Series({
    'Mobile': sum(converted_df[f'{col}_type'] == 'Mobile' for col in phone_cols),
    'Landline': sum(converted_df[f'{col}_type'] == 'Landline' for col in phone_cols),
    'Other/Unknown': sum(converted_df[f'{col}_type'].isin(['Unknown', 'Other']) for col in phone_cols)
})
phone_type_combined.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightgray'])
plt.title('Distribution of Phone Number Types')
plt.ylabel('')
plt.tight_layout()
plt.savefig('au_phone_distribution.png')
print("Saved phone type distribution plot to 'au_phone_distribution.png'")

print("\nExample complete!")