import os
import json
import logging
import boto3
import chromadb
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from chromadb.utils.embedding_functions import EmbeddingFunction

logger = logging.getLogger(__name__)

class BedrockEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using AWS Bedrock Titan V2."""
    def __init__(self, session: boto3.Session, region_name: str = Config.AWS_REGION):
        self.client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = Config.TITAN_EMBED_MODEL_ID

    def __call__(self, input: List[str]) -> List[List[float]]:
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
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 1024)
        return embeddings

class KnowledgeBase:
    """Handles the Vector Database (ChromaDB), Embeddings, and Document Ingestion."""
    def __init__(self, session: boto3.Session, region_name: str = Config.AWS_REGION):
        self.embedding_fn = BedrockEmbeddingFunction(session, region_name)
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def ingest_text(self, text: str, metadata: Dict[str, Any] = None):
        """Chunk and store text in the vector database."""
        chunks = self.text_splitter.split_text(text)
        ids = [f"chunk_{os.urandom(4).hex()}" for _ in range(len(chunks))]
        metadatas = [metadata or {} for _ in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(chunks)} chunks into ChromaDB.")
        return len(chunks)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str):
        """Extract text from PDF and ingest it."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        return self.ingest_text(full_text, {"filename": filename, "type": "pdf"})

    def clear_database(self):
        """Delete all documents from the collection."""
        try:
            # Get all IDs and delete them
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            logger.info("Knowledge base cleared.")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def retrieve(self, query: str, n_results: int = 2) -> str:
        """Search the knowledge base and return combined context with source info."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i in range(len(documents)):
            filename = metadatas[i].get("filename", "Unknown File")
            text = documents[i][:800].encode('ascii', 'ignore').decode('ascii')
            context_parts.append(f"[Source: {filename}]\n{text}")
            
        context = "\n---\n".join(context_parts)
        logger.info(f"Retrieved context for query: {query}")
        return context
