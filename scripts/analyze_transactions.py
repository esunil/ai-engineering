import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Dataset Generation
# ---------------------------------------------------------
transactions = [
    "CHEVRON GAS STN 4829 AUSTIN TX",            # Fuel purchase
    "EXXONMOBIL FUEL PAY AT PUMP",               # Fuel purchase (variant)
    "NETFLIX DIGITAL SUBSCRIPTION MO CHARGE",    # Recurring entertainment
    "NETFLIX.COM CARD PURCHASE 888-638-3549",    # Recurring entertainment (variant)
    "INTERNAL TRANSFER TO CHECKING ACCT 9482",   # Bank transfer
    "VENMO CASHOUT FROM JOHN DOE",               # Peer-to-peer payment
    "ZELLE PAYMENT FROM J SMITH"                 # Peer-to-peer payment (variant)
]

print(f"Loaded {len(transactions)} transaction records for analysis...\n")

# ---------------------------------------------------------
# 2. Generate Embeddings
# ---------------------------------------------------------
# Loading a lightweight, highly accurate sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(transactions)

print(f"Generated embeddings shape: {embeddings.shape}")
print(f"Dimensions per vector: {embeddings.shape[1]}\n")

# ---------------------------------------------------------
# 3. Method A: Pairwise Calculation with scikit-learn
# ---------------------------------------------------------
# Computes the full NxN matrix efficiently
sklearn_matrix = cosine_similarity(embeddings)

print("--- Scikit-Learn Pairwise Similarity Matrix ---")
print(np.round(sklearn_matrix, 3))
print("\n" + "="*50 + "\n")
