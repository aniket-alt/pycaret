import pandas as pd
from pycaret.arules import * # <-- CHANGE 1: We import from .arules

print("--- Task 6: Association Rules (Market Basket) ---")

# 1. LOAD THE DATA (Manual Way)
try:
    # This file has special characters, so we need 'encoding'
    df = pd.read_csv('Online Retail.csv', encoding='ISO-8859-1')
    print("Successfully loaded Online Retail.csv")
except FileNotFoundError:
    print("Error: 'Online Retail.csv' not found.")
    print("Please download it from Kaggle and place it in the same folder.")
    exit()

# This file is LARGE. Let's sample 20,000 rows.
df = df.sample(n=20000, random_state=123)

# Data Cleaning: We MUST drop rows with no InvoiceNo
df = df.dropna(subset=['InvoiceNo'])
print(f"Using a sample of {len(df)} transactions.")


# 2. RUN THE INCANTATION (The Setup)
print("Setting up PyCaret experiment...")
#
# !! CRITICAL CHANGE !!
# 'setup' is different. We MUST tell it the transaction and item columns.
# There is no session_id, no use_gpu.
#
# This setup function doesn't return anything. It just preps the data.
setup(data=df, 
      transaction_id='InvoiceNo', # <-- Column with the "receipt number"
      item_id='Description')   # <-- Column with the "item name"
print("Setup complete.")


# 3. CREATE A MODEL (Find the rules)
print("Finding association rules...")
# We ask it to find rules with a minimum 'support' (how often
# the items appear together). 0.02 means 2% of all transactions.
rules = create_model(metric='support', threshold=0.02)
print("Rules found:")
print(rules)


# 4. FULFILL THE REQUIREMENT (Workaround for 'get_config')
print("\n--- Fulfilling 'get_config' Requirement ---")
#
# The 'pycaret.arules' module does not have a get_config() function.
# As a workaround, we will print the key configuration details.
#
print(f"Data Shape: {df.shape}")
print(f"Transaction ID Column: 'InvoiceNo'")
print(f"Item ID Column: 'Description'")


print("\n--- Task 6 Complete ---")