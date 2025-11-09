import pandas as pd
from pycaret.regression import * # <-- CHANGE 1: We import from .regression
from pycaret.datasets import get_data

print("--- Task 3: Regression (Predicting Medical Charges) ---")

# 1. LOAD THE DATA
df = get_data('insurance') # <-- CHANGE 2: We load the 'insurance' dataset
print("Successfully loaded built-in Insurance dataset.")

# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
# PyCaret will see 'charges' is a number (a 'numeric' type)
# and automatically start a regression experiment.
reg_session = setup(data=df, 
                    target='charges', # <-- CHANGE 3: Our target is now 'charges'
                    session_id=123, 
                    use_gpu=False) 
print("Setup complete.")


# 3. RUN THE TOURNAMENT (Compare Models)
print("Comparing models to find the best one...")
best_model = compare_models()
print("Model comparison complete.")
print("\nBest Model Found:")
print(best_model)

# 4. FULFILL THE REQUIREMENT (The 'get_config' orb)
print("\n--- Fulfilling 'get_config' Requirement ---")

X_train_processed = get_config('X_train')
print("Shape of preprocessed X_train from get_config('X_train'):")
print(X_train_processed.shape)

target_name = get_config('target_param')
print(f"Target variable from get_config('target_param'): {target_name}")


print("\n--- Task 3 Complete ---")