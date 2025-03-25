"""
Test for the optimized topic modeling functionality with pickling support.

This script tests the create_topic_model_optimized function and ensures
that the results can be properly pickled and unpickled.
"""

import pandas as pd
import numpy as np
from freamon.utils.text_utils import create_topic_model_optimized
import pickle
import multiprocessing

# Create test dataframe with text data
texts = [
    "Economics and finance are important fields of study. Markets fluctuate based on economic indicators.",
    "Climate change is affecting global weather patterns and causing more extreme events.",
    "The financial markets showed signs of recovery after the economic downturn.",
    "Medical research has led to important breakthroughs in disease treatment.",
    "Technology companies are leading innovation in artificial intelligence.",
    "Healthcare systems worldwide face challenges with aging populations.",
    "Entertainment industry revenues have increased with streaming services.",
    "Sports events draw large crowds and generate significant revenue.",
    "Renewable energy sources are becoming more cost-effective and widespread.",
    "Educational systems are evolving to incorporate more technology.",
    "Science and technology drive economic growth through innovation.",
    "The banking sector implements new regulations after financial crises.",
    "Media coverage influences public opinion on political matters.",
    "Transportation infrastructure requires significant investment.",
    "Economic policy affects business sentiment and investment decisions."
]

# Create dataframe
df = pd.DataFrame({'text': texts})

print("Testing optimized topic modeling with pickling support...")

# Set up multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

# Run optimized topic modeling
result = create_topic_model_optimized(
    df=df,
    text_column='text',
    n_topics='auto',
    auto_topics_range=(2, 5),
    method='nmf',
    use_multiprocessing=True
)

# Print discovered topics
print(f"\nDiscovered {result['topic_model']['n_topics']} topics:")
for topic_idx, words in result['topics']:
    print(f"Topic {topic_idx+1}: {', '.join(words[:5])}")

# Test pickling
print("\nTesting pickling support...")
try:
    # Create a temporary file
    pickle_file = "temp_model.pkl"
    
    # Pickle the model
    with open(pickle_file, 'wb') as f:
        pickle.dump(result, f)
    
    # Unpickle the model
    with open(pickle_file, 'rb') as f:
        loaded_result = pickle.load(f)
    
    # Check if the unpickled model has the same topics
    assert loaded_result['topic_model']['n_topics'] == result['topic_model']['n_topics']
    print("✓ Pickling and unpickling successful")
    
    # Clean up
    import os
    os.remove(pickle_file)
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"✗ Pickling failed: {str(e)}")
    raise