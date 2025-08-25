# flightdashboard

# ✈️ Flight Scheduling Pro

An **AI-powered flight operations dashboard** that lets users query flight schedules in natural language and simulate the cascading impacts of flight delays.  
Built with a hybrid approach combining **DuckDB, FAISS, NetworkX, Streamlit, and Google Gemini**.  

---

## 🚀 Features

- **Natural Language Queries**  
  Ask questions like *“Show me all flights for Air India”* or *“What is the average delay per airline?”*.  
  The system auto-generates and executes SQL queries against the flight database.  

- **Delay Propagation Simulation**  
  Run *“What if”* scenarios such as *“What happens if flight AI101 is delayed by 60 minutes?”*  
  → The system simulates cascading delays across connected flights using an aircraft rotation graph.  

- **LLM-powered Tool Selection**  
  Google **Gemini** decides whether to run a SQL query or a delay simulation based on your question.  

- **Interactive Dashboard**  
  Built with **Streamlit**, results are visualized as clean tables and explanations.  

---

## 🧩 Tech Stack

- **Frontend / Dashboard:** [Streamlit](https://streamlit.io/)  
- **Database:** [DuckDB](https://duckdb.org/)  
- **Vector Search Engine:** [FAISS](https://github.com/facebookresearch/faiss)  
- **Language Model:** [Google Gemini API](https://ai.google.dev/)  
- **Graph Processing:** [NetworkX](https://networkx.org/)  
- **Data Handling:** Pandas, NumPy  
- **Environment Management:** python-dotenv  
- **Serialization:** Pickle  

---

## ⚙️ System Workflow

### 1. **Knowledge Base Creation (`embed_schema.py`)**
- Loads `detailed_flights.csv` into DuckDB (`flights.db`).
- Extracts schema information (table & columns).
- Embeds schema details using Gemini → stored in **FAISS index**.
- Builds an **aircraft rotation graph** where:
  - Nodes = flights  
  - Edges = consecutive flights for the same aircraft  
- Saves:  
  - `schema.index` → schema embeddings  
  - `schema_mapping.pkl` → schema descriptions  
  - `rotations_graph.gpickle` → rotation graph  

### 2. **Main Dashboard (`app.py`)**
- Loads FAISS, schema mapping, graph, and Gemini model.
- User enters a natural language query:
  1. Converts query into embedding and retrieves schema context.
  2. Gemini analyzes and returns a JSON decision:  
     - **SQL Query** → runs in DuckDB.  
     - **Delay Simulation** → propagates delay in graph.  
  3. Results displayed in an interactive Streamlit UI.  

---

## 📊 Example Queries
- *“Show me all flights for Air India”*  
- *“What is the average delay per airline?”*  
- *“Count the number of flights per day”*  
- *“What if flight 20250823_56_6E531 is delayed by 45 minutes?”*  
- *“If flight 20250822_64_AI101 has an extra delay of 60 minutes, what happens?”*  

---

## 📂 Project Structure
flightdashboard/
│── app.py # Streamlit dashboard
│── embed_schema.py # Prepares schema knowledge base and rotation graph
│── requirements.txt # Dependencies
│── flights.db # DuckDB database (auto-generated)
│── detailed_flights.csv # Input flight dataset
│── schema.index # FAISS index of schema embeddings
│── schema_mapping.pkl # Schema description mappings
│── rotations_graph.gpickle # Flight rotation graph
│── .env # Contains GEMINI_API_KEY
│── venv/ # Virtual environment (optional)

---

## 🛠️ Setup & Installation
**Clone the Repository**
   git clone https://github.com/Rishit1378/flightdashboard.git
   cd flightdashboard

**Create Virtual Environment**
  python -m venv venv
  source venv/bin/activate   # Mac/Linux
  venv\Scripts\activate      # Windows
  
**Install Dependencies**
  pip install -r requirements.txt

**Set Up Environment Variables**
  Create a .env file in the root directory:
  GEMINI_API_KEY=your_api_key_here

**Prepare Knowledge Base**
  python embed_schema.py

**Run the Application**
  streamlit run app.py

**Requirements**
Python 3.9+
Google Gemini API key
Dependencies listed in requirements.txt:
  streamlit
  pandas
  duckdb
  google-generativeai
  faiss-cpu
  python-dotenv
  networkx

**🔮 Future Enhancements**
  Visualization of delay propagation using network graphs.
  Support for multi-airline / multi-aircraft complex schedules.
  Integration with real-time flight data APIs.
  Advanced natural language support for multi-step queries.
