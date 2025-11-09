import pandas as pd
from pycaret.classification import *

print("--- Task 1: Binary Classification (Titanic) ---")

# 1. LOAD THE DATA

try:
    df = pd.read_csv('titanic_train.csv')
    print("Successfully loaded Titanic titanic_train.csv")
except FileNotFoundError:
    print("Error: 'titanic_train.csv' not found.")
    print("Please download it from Kaggle and place it in the same folder.")
    exit()

# Basic data pre-processing
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print("Data pre-processing: Dropped unnecessary columns.")

# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")


clf_session = setup(data=df, 
                    target='Survived', 
                    session_id=123, 
                    use_gpu=False) 
print("Setup complete.")


# 3. RUN THE TOURNAMENT (Compare Models)
print("Comparing models to find the best one (using CPU)...")
best_model = compare_models()
print("Model comparison complete.")
print("\nBest Model Found:")
print(best_model)


print("\n--- Fulfilling 'get_config' Requirement ---")

# Get the preprocessed training data
X_train_processed = get_config('X_train')
print("Shape of preprocessed X_train from get_config('X_train'):")
print(X_train_processed.shape)

session_id = get_config('session_id')
print(f"Session ID from get_config('session_id'): {session_id}")


print("\n--- Task 1 Complete ---")