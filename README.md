# GenAI-Powered Financial Knowledge Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) application built with **LangChain**, **ChromaDB**, and **Groq Cloud Infrastructure** to enable real-time, context-aware question answering over dense financial filings (e.g., SEC 10-K/10-Q reports).

## 🚀 Key Features & Architectural Highlights
* **Advanced Semantic Search:** Indexes and processes **500+ document chunks** using local open-source embeddings (`all-MiniLM-L6-v2`), eliminating embedding API costs.
* **Ultra-Low Latency Inference:** Powered by **Groq's LPU Inference Engine** (utilizing Llama 3.3/3.1 models) to deliver response synthesis under 1.5 seconds.
* **Citation & Explainability Engine:** Built a custom metadata-injection prompt framework that forces the LLM to attribute metrics to specific source files and page numbers, eliminating hallucination risks.
* **Interactive Analytics UI:** Developed an intuitive **Streamlit** dashboard featuring context-window customization (Top-K tuning) and real-time inference latency tracking.

## 🛠️ The Tech Stack
* **Orchestration:** LangChain / LangChain-Classic
* **Vector Database:** ChromaDB
* **LLM Engine:** Groq Cloud (Llama 3.3 70B / Llama 3.1 8B)
* **Embeddings:** Hugging Face Sentence-Transformers
* **Frontend:** Streamlit

## 📦 System Architecture
1. **Ingestion (`ingest.py`):** Parses raw financial PDFs via `PyPDFDirectoryLoader` ➡️ Chunks text recursively (1000-char window, 200-char overlap) ➡️ Generates text-vector embeddings ➡️ Persists data locally in a structured ChromaDB collection.
2. **Execution (`app.py`):** Runs the Streamlit UI server ➡️ Fetches query matching tokens from Chroma via Cosine Similarity retrieval ➡️ Pumps formatted page-metadata context into the Llama context pipeline ➡️ Renders the response alongside dynamic source drop-down blocks.

## 🔧 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/financial-rag-assistant.git](https://github.com/YOUR_GITHUB_USERNAME/financial-rag-assistant.git)
   cd financial-rag-assistant

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate 
```
3. Install dependencies:

```bash
pip install -r requirements.txt # On Windows use: venv\Scripts\activate
```
4. Configure Environment Variables:

Create a .env file in the root directory and insert your Groq API key:

GROQ_API_KEY=your_groq_api_key_here

5.Run Document Ingestion:

Place your raw corporate financial PDFs inside the data/ directory, then process them: 

```bash
python ingest.py 
```

6. Launch the Web Dashboard: 

```bash
streamlit run app.py
```

