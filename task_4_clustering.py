import pandas as pd
# from pycaret.datasets import get_data  <-- We don't need this anymore
from pycaret.clustering import *

print("--- Task 4: Clustering (Mall Customer Segments) ---")

# 1. LOAD THE DATA (Manual Way)
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("Successfully loaded Mall_Customers.csv")
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found.")
    print("Please download it from Kaggle and place it in the same folder.")
    exit()

# (Optional but good practice: drop the ID column)
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
#
# !! CRITICAL CHANGE !!
# There is NO 'target' parameter! This is unsupervised learning.
#
clu_session = setup(data=df, 
                    session_id=123, 
                    use_gpu=False) 
print("Setup complete.")


# 3. CREATE A MODEL (No 'compare_models()')
print("Creating a K-Means model with 4 clusters...")
kmeans_model = create_model('kmeans', num_clusters=4)
print(kmeans_model) # This prints the "scoreboard"


# 4. ASSIGN THE CLUSTERS
print("\nAssigning clusters back to the original data...")
results = assign_model(kmeans_model)
print("Data with new 'Cluster' column:")
print(results.head())


# 5. FULFILL THE REQUIREMENT (The 'get_config' orb)
print("\n--- Fulfilling 'get_config' Requirement ---")

X_processed = get_config('X') # 'X' is all the data
print("Shape of preprocessed data from get_config('X'):")
print(X_processed.shape)

#
# --- THIS IS THE FIX ---
# In clustering, the session_id is stored as 'seed'
#
seed = get_config('seed')
print(f"Seed from get_config('seed'): {seed}") # Should print 123


print("\n--- Task 4 Complete ---")