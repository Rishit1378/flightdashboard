# app.py
import streamlit as st
import pandas as pd
import duckdb
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pickle
import networkx as nx
import json

# --- Page and Gemini Configuration ---
st.set_page_config(layout="wide", page_title="Flight Scheduling Pro")
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    st.error("Please create a .env file and add your GEMINI_API_KEY.")
    st.stop()

# --- File Paths and Model Names ---
DB_FILE = "flights.db"
FAISS_INDEX_PATH = "schema.index"
SCHEMA_MAPPING_PATH = "schema_mapping.pkl"
GRAPH_PATH = "rotations_graph.gpickle"
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "gemini-1.5-pro-latest"

# --- Caching and Resource Loading ---
@st.cache_resource
def load_resources():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(SCHEMA_MAPPING_PATH, 'rb') as f:
            schema_mapping = pickle.load(f)
        
        # --- CHANGE IS HERE ---
        # Old line (nx.read_gpickle) is replaced with pickle.load
        with open(GRAPH_PATH, 'rb') as f:
            graph = pickle.load(f)
        # --- END OF CHANGE ---

        model = genai.GenerativeModel(GENERATION_MODEL)
        return index, schema_mapping, graph, model
    except FileNotFoundError:
        st.error("Knowledge base files not found. Please run 'python embed_schema.py' first.")
        st.stop()

# --- Backend Logic ---
def get_embedding(text):
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="RETRIEVAL_QUERY")
    return np.array([result['embedding']], dtype='float32')

def propagate_delay(graph, all_flights_df, starting_flight, delay_minutes):
    """Simulates the cascading effect of a delay."""
    if starting_flight not in graph:
        return f"Error: Flight '{starting_flight}' not found in the rotation graph."
    
    affected_flights = []
    total_propagated_delay = 0
    
    queue = [(starting_flight, delay_minutes)]
    visited = {starting_flight}

    while queue:
        current_flight, incoming_delay = queue.pop(0)
        
        flight_details_series = all_flights_df[all_flights_df['flight_id'] == current_flight]
        if flight_details_series.empty:
            continue # Skip if flight details aren't found
        flight_details = flight_details_series.iloc[0]

        original_delay = flight_details['delay_minutes']
        new_total_delay = original_delay + incoming_delay

        affected_flights.append({
            "Flight ID": current_flight,
            "Route": flight_details['route'],
            "Original Delay (min)": original_delay,
            "New Total Delay (min)": new_total_delay
        })
        
        for next_flight in graph.successors(current_flight):
            if next_flight not in visited:
                visited.add(next_flight)
                queue.append((next_flight, new_total_delay))
                total_propagated_delay += new_total_delay
    
    return affected_flights, total_propagated_delay

def get_gemini_decision(user_question, context_str):
    """Gets Gemini to decide which tool to use and provides parameters."""
    prompt = f"""
    You are an intelligent router for a flight operations dashboard. Your task is to analyze the user's question and decide which tool to use.
    You have two tools available:
    1. "sql_query": Use this for questions about existing data (e.g., "what is", "show me", "count", "average").
    2. "delay_simulation": Use this for hypothetical "what-if" questions about delays.

    **Database Schema Context:**
    {context_str}

    **User Question:**
    "{user_question}"

    **Instructions:**
    Respond with a single JSON object.
    - If using "sql_query", the JSON should be: {{"tool": "sql_query", "query": "<DUCKDB_SQL_QUERY>"}}
    - If using "delay_simulation", the JSON should be: {{"tool": "delay_simulation", "flight_id": "<FLIGHT_ID>", "delay_minutes": <DELAY_IN_MINUTES>}}
    - Extract the flight_id and delay_minutes accurately.
    """
    
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    # A simple loop to clean Gemini's response, removing markdown backticks
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_response)

# --- Main Application Logic ---
st.title("✈️ Flight Scheduling Pro")
st.write("An intelligent assistant for querying flight data and simulating delay impacts.")

index, schema_mapping, graph, model = load_resources()
con = duckdb.connect(DB_FILE, read_only=True)
all_flights_df = con.execute("SELECT * FROM FLIGHTS").fetchdf()
con.close()

example_queries = [
    "What if flight 20250823_56_6E531 is delayed by 45 more minutes?",
    "Show me all flights for Air India",
    "What is the average delay for each airline?",
    "If flight 20250822_64_AI101 has an extra delay of 60 minutes, what happens?",
    "Count the number of flights per day"
]
selected_query = st.selectbox("Choose an example or type your own below:", options=example_queries)
user_input = st.text_input("Your Question:", selected_query)

if st.button("Get Answer"):
    if not user_input:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your question and routing to the correct tool..."):
            question_embedding = get_embedding(user_input)
            D, I = index.search(question_embedding, k=5)
            context_str = "\n".join([schema_mapping[i] for i in I[0]])
            
            try:
                decision = get_gemini_decision(user_input, context_str)
                tool_to_use = decision.get("tool")

                if tool_to_use == "sql_query":
                    st.info("Tool chosen: **SQL Query**")
                    sql_query = decision.get("query")
                    with st.expander("Generated SQL Query"):
                        st.code(sql_query, language="sql")
                    con = duckdb.connect(DB_FILE, read_only=True)
                    result_df = con.execute(sql_query).fetchdf()
                    con.close()
                    st.dataframe(result_df)

                elif tool_to_use == "delay_simulation":
                    st.info("Tool chosen: **Delay Simulation**")
                    flight_id = decision.get("flight_id")
                    delay_minutes = int(decision.get("delay_minutes"))
                    
                    st.write(f"Simulating a **{delay_minutes} minute** extra delay for flight **{flight_id}**...")
                    
                    result, total_delay = propagate_delay(graph, all_flights_df, flight_id, delay_minutes)
                    
                    if isinstance(result, str):
                        st.error(result)
                    else:
                        result_df = pd.DataFrame(result)
                        st.dataframe(result_df)
                        st.success(f"Total propagated delay to subsequent flights: **{total_delay} minutes**.")

                else:
                    st.error("Sorry, I could not determine the correct tool to use for your question.")

            except (json.JSONDecodeError, AttributeError, Exception) as e:
                st.error(f"An error occurred while processing your request. The model may have returned an invalid response. Details: {e}")