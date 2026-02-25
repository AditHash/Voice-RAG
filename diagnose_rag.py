import os
import json
import logging
import boto3
import fitz
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from src.core.config import settings
from src.core.auth import get_aws_session

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Replicating BedrockEmbeddingFunction ---
class BedrockEmbeddingFunction(EmbeddingFunction):
    def __init__(self, session: boto3.Session):
        self.client = session.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self.model_id = settings.TITAN_EMBED_MODEL_ID

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({"inputText": text})
                )
                response_body = json.loads(response.get("body").read())
                embeddings.append(response_body.get("embedding"))
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                # Fallback to zero vector if embedding fails (Titan v2 typically has 1024 dims)
                embeddings.append([0.0] * 1024) 
        return embeddings

# --- Replicating KnowledgeBaseService Core Logic ---
class DiagnosticKnowledgeBase:
    def __init__(self, session: boto3.Session):
        self.embedding_fn = BedrockEmbeddingFunction(session)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def ingest_pdf_manual(self, pdf_path: Path):
        logger.info(f"Step 1: Loading PDF from {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            logger.info(f"  - Extracted text from Page {page_num + 1}, length: {len(page_text)}")
            full_text += page_text + "

" # Add separators between pages

        logger.info(f"Step 2: Total extracted text length: {len(full_text)}")
        
        logger.info(f"Step 3: Chunking text with chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP}")
        chunks = self.text_splitter.split_text(full_text)
        logger.info(f"  - Created {len(chunks)} chunks.")

        ids = [f"diag_chunk_{os.urandom(4).hex()}" for _ in range(len(chunks))]
        metadatas = [{"filename": pdf_path.name, "type": "pdf", "source_chunk_idx": i} for i in range(len(chunks))]
        
        logger.info("Step 4: Generating embeddings and adding to ChromaDB...")
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        logger.info(f"Step 5: Successfully ingested {len(chunks)} chunks into ChromaDB.")
        return len(chunks)

    def retrieve_manual(self, query: str):
        logger.info(f"Step 6: Performing retrieval for query: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=settings.NUM_RETRIEVAL_RESULTS, # Assuming a new setting for diagnosis
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            logger.warning("  - No documents found in retrieval.")
            return "No relevant information found."
        
        logger.info(f"Step 7: Retrieved {len(documents)} results:")
        context_parts = []
        for i in range(len(documents)):
            filename = metadatas[i].get("filename", "Unknown")
            source_idx = metadatas[i].get("source_chunk_idx", "N/A")
            distance = distances[i]
            text = documents[i][:500] # Show first 500 chars for brevity

            logger.info(f"  - Result {i+1}: Source='{filename}', Chunk='{source_idx}', Distance={distance:.4f}
    Text: '{text.replace('n', ' ').replace('t', ' ')}'")
            context_parts.append(f"[Source: {filename}, Dist: {distance:.2f}]
{documents[i]}")
            
        return "
---
".join(context_parts)

    def clear_all_manual(self):
        logger.info("Step X: Clearing entire ChromaDB collection...")
        try:
            # Need to re-create collection if clearing all in PersistentClient
            self.chroma_client.delete_collection(name=settings.COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info("  - ChromaDB collection cleared and re-created.")
            return True
        except Exception as e:
            logger.error(f"  - Error clearing database: {e}")
            return False

async def main():
    # --- Configuration and Session ---
    session = get_aws_session()
    
    # --- Initialize Diagnostic KB ---
    diag_kb = DiagnosticKnowledgeBase(session)
    
    # --- Clear previous data ---
    diag_kb.clear_all_manual()

    # --- Ingestion ---
    # !!! IMPORTANT !!! Replace 'sample.pdf' with the actual path to your PDF file
    pdf_file_path = Path("sample.pdf") # <--- MAKE SURE THIS PDF EXISTS IN YOUR PROJECT ROOT
    if not pdf_file_path.exists():
        logger.error(f"Error: PDF file not found at {pdf_file_path}. Please create one or update the path.")
        return

    ingested_chunks = diag_kb.ingest_pdf_manual(pdf_file_path)
    if ingested_chunks == 0:
        logger.error("No chunks ingested. RAG will not work.")
        return

    # --- Retrieval ---
    test_queries = [
        "What is this document about?",
        "What are the main topics?",
        "Tell me about the key figures.",
        "What dates are mentioned?",
        "Summarize the content."
    ]

    for query in test_queries:
        logger.info("
" + "="*50 + f"
TESTING QUERY: '{query}'")
        retrieved_context = diag_kb.retrieve_manual(query)
        logger.info(f"
Retrieved Context for '{query}':
{retrieved_context}")
        logger.info("="*50 + "
")

if __name__ == "__main__":
    # Add a setting for NUM_RETRIEVAL_RESULTS if not already in config.py
    # This is temporary for diagnosis, or you can add it to settings.
    settings.NUM_RETRIEVAL_RESULTS = 5 # Set a diagnostic number of results
    asyncio.run(main())
