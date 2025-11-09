import pandas as pd
from pycaret.time_series import * # <-- CHANGE 1: We import from .time_series
from pycaret.datasets import get_data 

print("--- Task 7: Time Series Forecasting (Airline) ---")

# 1. LOAD THE DATA
# We will try loading the built-in dataset first.
# If this fails, it's the firewall, and we'll switch to a manual CSV.
try:
    df = get_data('airline')
    print("Successfully loaded built-in Airline dataset.")
    print(df.head())
except ValueError:
    print("\n" + "="*50)
    print("ERROR: Could not download 'airline' dataset due to firewall.")
    print("Please tell me you got this error, and I will give you the manual CSV-based script.")
    print("="*50 + "\n")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
#
# !! CRITICAL CHANGE !!
# We MUST specify a 'fh' (Forecasting Horizon).
# We'll forecast the next 12 steps (12 months).
# The built-in data is already indexed, so we don't need 'target'.
#
ts_session = setup(data=df, 
                   fh=12, # Forecast 12 steps (months) ahead
                   fold=3, # Use 3 folds for faster cross-validation (it's faster)
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

# In time series, 'y_train' is the key training data
y_train_data = get_config('y_train')
print("Shape of training data from get_config('y_train'):")
print(y_train_data.shape)

seed = get_config('seed')
print(f"Seed from get_config('seed'): {seed}")


print("\n--- Task 7 Complete ---")