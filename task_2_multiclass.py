import pandas as pd
from pycaret.classification import *
from pycaret.datasets import get_data # <-- New! To load a built-in dataset

print("--- Task 2: Multiclass Classification (Iris) ---")

# 1. LOAD THE DATA
# We are loading a built-in dataset this time.
df = get_data('iris')
print("Successfully loaded built-in Iris dataset.")
print("Target classes:", df['species'].unique())

# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
# PyCaret is smart. It will see 'species' has 3 unique values
# and automatically start a multiclass classification experiment.
clf_session = setup(data=df, 
                    target='species', # <-- New target column
                    session_id=123, 
                    use_gpu=False) # <-- Still False for your local machine
print("Setup complete.")


# 3. RUN THE TOURNAMENT (Compare Models)
print("Comparing models to find the best one...")
best_model = compare_models()
print("Model comparison complete.")
print("\nBest Model Found:")
print(best_model)

# 4. FULFILL THE REQUIREMENT (The 'get_config' orb)
print("\n--- Fulfilling 'get_config' Requirement ---")

# Get the preprocessed training data
X_train_processed = get_config('X_train')
print("Shape of preprocessed X_train from get_config('X_train'):")
print(X_train_processed.shape)

# Get the target column name
target_name = get_config('target_param')
print(f"Target variable from get_config('target_param'): {target_name}")


print("\n--- Task 2 Complete ---")