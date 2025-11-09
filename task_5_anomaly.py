import pandas as pd
from pycaret.anomaly import * # <-- CHANGE 1: We import from .anomaly

print("--- Task 5: Anomaly Detection (Fraud) ---")

# 1. LOAD THE DATA (Manual Way)
try:
    df = pd.read_csv('creditcard.csv')
    print("Successfully loaded creditcard.csv")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found.")
    print("Please download it from Kaggle and place it in the same folder.")
    exit()

#
# !! CRITICAL STEP !!
# This file is HUGE (280,000+ rows). Running on the full file will be slow.
# Let's take a random sample of 10,000 rows to speed things up.
#
df = df.sample(n=10000, random_state=123)
print(f"Using a random sample of {len(df)} rows.")

# In a real-world task, we wouldn't have the 'Class' (fraud/not fraud) column.
# We're trying to FIND the fraud. So let's drop it.
if 'Class' in df.columns:
    df = df.drop('Class', axis=1)

# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
#
# Like clustering, there is NO 'target' parameter.
#
ano_session = setup(data=df, 
                    session_id=123, 
                    use_gpu=False) 
print("Setup complete.")


# 3. CREATE A MODEL (No 'compare_models()')
# We create a specific type of anomaly detection model.
# 'iforest' (Isolation Forest) is a very popular and fast choice.
print("Creating an Isolation Forest model...")
iforest_model = create_model('iforest')
print(iforest_model) # This prints a simple confirmation


# 4. ASSIGN THE ANOMALIES
# Now we use our model to "score" every transaction.
print("\nAssigning anomaly scores to the data...")
results = assign_model(iforest_model)
print("Data with new 'Anomaly' and 'Anomaly_Score' columns:")
print(results.head())

# Let's see the ones our model flagged as "most weird"
print("\nTop 5 most likely anomalies (highest score):")
print(results.sort_values(by='Anomaly_Score', ascending=False).head())


# 5. FULFILL THE REQUIREMENT (The 'get_config' orb)
print("\n--- Fulfilling 'get_config' Requirement ---")

X_processed = get_config('X')
print("Shape of preprocessed data from get_config('X'):")
print(X_processed.shape)

seed = get_config('seed')
print(f"Seed from get_config('seed'): {seed}")


print("\n--- Task 5 Complete ---")