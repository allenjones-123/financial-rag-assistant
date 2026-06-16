import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def run_pipeline():
    print("🚀 Starting Financial Document Ingestion Pipeline...")
    
    # 1. Document Ingestion
    # Automatically searches the /data directory and loads any PDFs found
    if not os.path.exists("data") or not os.listdir("data"):
        print("❌ Error: 'data/' directory is missing or empty. Drop your financial PDFs there first.")
        return
        
    loader = PyPDFDirectoryLoader("data")
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} individual pages from source documents.")
    
    # 2. Optimized Chunking Strategy
    # Using recursive splitting to keep financial paragraphs and tables intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✂️  Successfully split pages into {len(chunks)} text chunks.")
    
    # Injecting custom metadata tracking for filtering optimizations
    # Automatically extracts company name from filename as a metadata tag
    for chunk in chunks:
        source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
        company_name = source_file.split("_")[0] if "_" in source_file else "General"
        chunk.metadata["company"] = company_name.upper()

    # 3. Embedding Generation (100% Free / Local)
    print("🧬 Initializing Embedding Engine (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Vector Storage Generation & Persistence
    print("💾 Indexing text vectors and saving to ChromaDB database...")
    db_storage_path = "./chroma_db"
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_storage_path
    )
    
    print(f"✅ Success! Your Vector DB is compiled with {vectorstore._collection.count()} chunks.")
    print(f"📁 Database files securely saved under '{db_storage_path}/'")

if __name__ == "__main__":
    run_pipeline()