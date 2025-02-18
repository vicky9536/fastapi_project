import os
import logging
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from config import LANGCHAIN_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
embedder = HuggingFaceEmbeddings(model_name=LANGCHAIN_MODEL_PATH)

# Define index location
VECTOR_STORE_PATH = "vector_store"

# Try loading an existing FAISS index, or create a new one
try:
    faiss_index = FAISS.load_local(VECTOR_STORE_PATH, embedder)
    logger.info("Loaded existing FAISS index.")
except:
    faiss_index = FAISS.from_texts([], embedder)
    logger.info("Created a new FAISS index.")

llm = OpenAI(model_name="gpt-4o")

# Create a RetrievalQA chain (connects FAISS search with LLM)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=faiss_index.as_retriever(), chain_type="stuff"
)

def add_documents_from_files(file_paths: List[str]):
    """
    Extracts text from files (PDF, TXT), embeds, and stores them in FAISS.
    """
    global faiss_index
    try:
        documents = []

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            else:
                return {"error": f"Unsupported file format: {ext}"}
            
            docs = loader.load()
            documents.extend(docs)

        if not documents:
            return {"error": "No text extracted from files."}

        # Convert documents to FAISS embeddings
        faiss_index.add_documents(documents)
        faiss_index.save_local(VECTOR_STORE_PATH)
        logger.info(f"Added {len(documents)} documents from files.")
        
        return {"message": f"Successfully indexed {len(documents)} documents."}
    
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return {"error": str(e)}


def semantic_search(query: str, top_k: int = 5):

    try:
        if not query:
            raise ValueError("Query string is empty.")

        results = faiss_index.similarity_search(query, k=top_k)
        response = [{"content": r.page_content, "metadata": r.metadata} for r in results]
        return response
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return {"error": str(e)}


def answer_question(query: str):

    try:
        if not query:
            raise ValueError("Query string is empty.")

        answer = qa_chain.run(query)
        return {"query": query, "answer": answer}
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"error": str(e)}


def save_index():

    try:
        faiss_index.save_local(VECTOR_STORE_PATH)
        logger.info("FAISS index saved successfully.")
        return {"message": "Index saved successfully."}
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return {"error": str(e)}
