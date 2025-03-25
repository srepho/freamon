"""
Utility to disable matplotlib LaTeX parsing for currency symbols.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Configuring matplotlib...")
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

def plot_with_currency_values():
    """Create a test plot with currency values"""
    # Create test data
    data = [
        {'item': 'Sony TV', 'price': 'INSD quotes are $5,995.00 for FWD65A95L SONY 65" 4K UHD QD OLED TV'},
        {'item': 'LG TV', 'price': '$5,384.00 for OLED65C3'},
        {'item': 'Samsung TV', 'price': '$2,984.00 for QN65S95D 65" 4K UHD QD OLED TV'},
    ]
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot data
    ax.bar(df['item'], [1000, 2000, 3000])
    
    # Add currency values as text
    for i, row in enumerate(data):
        plt.text(i, 500, row['price'], ha='center', rotation=90, fontsize=8)
    
    plt.title("Sample Plot with Currency Values")
    plt.tight_layout()
    plt.savefig('currency_test_plot.png')
    print("Plot saved to 'currency_test_plot.png'")

if __name__ == "__main__":
    plot_with_currency_values()