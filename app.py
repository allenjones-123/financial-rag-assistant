import time
import streamlit as st
from dotenv import load_dotenv

import os
# FORCE pure-python protobuf implementation at the absolute entry point
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from dotenv import load_dotenv

# --- The Rest of Your LangChain & Ecosystem Imports Go Here ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# ... [Keep the rest of your app.py code exactly the same] ...

# LangChain & Ecosystem Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Load environment variables
load_dotenv()
# Safe Secret Routing Layer for both Cloud and Local environments
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # If st.secrets throws an error locally because secrets.toml doesn't exist,
    # we catch it and gracefully fall back to the .env file loaded above.
    pass

# Ensure API Key is available
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found. Please ensure it is set in your .env file locally or in your cloud secrets.")
    st.stop()

# --- Page Configuration & Styling ---
st.set_page_config(page_title="GenAI Financial Assistant", page_icon="📈", layout="wide")
st.title("📈 GenAI-Powered Financial Knowledge Assistant")
st.markdown("Query enterprise knowledge bases and financial filings with zero-latency semantic search.")
st.markdown("---")

# --- Sidebar Controls (Resume Proof Points) ---
st.sidebar.header("⚙️ Retrieval Optimization Settings")
top_k = st.sidebar.slider("Top-K Document Retrieval (Context chunks)", min_value=1, max_value=10, value=4)
model_choice = st.sidebar.selectbox("Inference Engine", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

# Clear Chat History Button
if st.sidebar.button("🧹 Clear Conversation History"):
    st.session_state.messages = []
    st.rerun()

# --- Resource Caching for Performance ---
@st.cache_resource
def initialize_vector_store():
    """Loads the local persistent ChromaDB collection with HuggingFace Embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize components
vectorstore = initialize_vector_store()
llm = ChatGroq(model=model_choice, temperature=0.1)

# --- Construct the Production RAG Chain ---
# Custom Prompt Engineering to enforce strict adherence to context and enforce citations
system_prompt = (
    "You are an expert financial analyst assistant. Answer the user's question using ONLY the provided context below. "
    "If the answer cannot be found or deduced from the context, say 'I cannot find the answer in the uploaded documents.' "
    "Do not attempt to make up facts.\n\n"
    "Context:\n{context}\n\n"
    "Crucial Instruction: Identify the page numbers provided in the context chunks and include them in your answer text "
    "when attributing financial facts."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# FIX 1: Custom document prompt that blends the page metadata into the raw text stream for the LLM
doc_prompt = PromptTemplate(
    input_variables=["page_content", "page"],
    template="--- [Context Chunk from Page {page}] ---\n{page_content}\n"
)

# Create Document Processing Chain & Combine with Retriever
question_answer_chain = create_stuff_documents_chain(
    llm, 
    prompt_template,
    document_prompt=doc_prompt
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Chat Interface Stateful Tracking ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat logs
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Verified Source Documents"):
                for src in msg["sources"]:
                    st.caption(src)

# --- Handle New User Queries ---
if user_query := st.chat_input("Ask a financial question (e.g., 'What was Tesla's total revenue in 2023?')"):
    
    # Display user's input
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Process through RAG Pipeline
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Analyzing financial layers & synthesizing response..."):
            # Start the performance clock
            start_time = time.time()
            
            # Execute full LangChain pipeline
            result = rag_chain.invoke({"input": user_query})
            
            # Stop the performance clock
            latency = time.time() - start_time
            
            answer = result["answer"]
            source_docs = result.get("context", []) 
            
            # Append the latency metric cleanly to the response text
            performance_metrics = f"\n\n⏱️ *Inference Latency: {latency:.2f} seconds*"
            response_placeholder.markdown(answer + performance_metrics)
            
            # Extract and de-duplicate unique source attributes for UI display
            unique_sources = []
            for doc in source_docs:
                meta = doc.metadata
                file_name = os.path.basename(meta.get("source", "Unknown File"))
                page_num = meta.get("page", "N/A")
                if page_num != "N/A":
                    page_num = int(page_num) + 1  # Convert 0-indexed pages to human-readable
                
                source_string = f"📄 Document: {file_name} | Page: {page_num}"
                if source_string not in unique_sources:
                    unique_sources.append(source_string)
            
            # Render sources dynamically in a clean accordion layout
            if unique_sources:
                with st.expander("🔍 Verified Source Documents"):
                    for src in unique_sources:
                        st.caption(src)
                        
            # Save assistant response state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": unique_sources
            })