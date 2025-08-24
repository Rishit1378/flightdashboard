# embed_schema.py
import duckdb
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import networkx as nx

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
DB_FILE = "flights.db"
CSV_FILE = "detailed_flights.csv" # Make sure this points to your detailed dataset
FAISS_INDEX_PATH = "schema.index"
SCHEMA_MAPPING_PATH = "schema_mapping.pkl"
GRAPH_PATH = "rotations_graph.gpickle"
EMBEDDING_MODEL = "models/embedding-001"

print("Starting knowledge base and graph creation process...")

# 1. Load data from CSV into DuckDB
df = pd.read_csv(CSV_FILE)
df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
con = duckdb.connect(DB_FILE)
con.execute("DROP TABLE IF EXISTS FLIGHTS;")
con.execute("CREATE TABLE FLIGHTS AS SELECT * FROM df;")

# 2. Get schema information
schema_info = con.execute("PRAGMA table_info('FLIGHTS');").fetchall()
text_chunks = [f"The table is named FLIGHTS."]
for col in schema_info:
    text_chunks.append(f"The FLIGHTS table has a column named '{col[1]}' of type {col[2]}.")

# 3. Get embeddings and save FAISS index
result = genai.embed_content(model=EMBEDDING_MODEL, content=text_chunks, task_type="RETRIEVAL_DOCUMENT")
embeddings_np = np.array(result['embedding'], dtype='float32')
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, FAISS_INDEX_PATH)
with open(SCHEMA_MAPPING_PATH, 'wb') as f:
    pickle.dump(text_chunks, f)
print("FAISS index and schema mapping saved.")

# 4. Build and save the aircraft rotation graph
print("Building aircraft rotation graph...")
G = nx.DiGraph()
df_sorted = df.sort_values('scheduled_departure')
for aircraft, group in df_sorted.groupby('aircraft_id'):
    flights = list(group['flight_id'])
    for i in range(len(flights) - 1):
        G.add_edge(flights[i], flights[i+1])

# --- CHANGE IS HERE ---
# Old line (nx.write_gpickle) is replaced with pickle.dump
with open(GRAPH_PATH, 'wb') as f:
    pickle.dump(G, f)
# --- END OF CHANGE ---

print(f"Rotation graph with {G.number_of_nodes()} flights and {G.number_of_edges()} rotations saved.")

con.close()
print("\nProcess complete.")