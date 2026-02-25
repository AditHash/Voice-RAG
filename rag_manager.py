import os
import json
import logging
import boto3
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class BedrockEmbeddingFunction:
    """Custom embedding function for ChromaDB using AWS Bedrock Titan V2."""
    def __init__(self, session: boto3.Session, region_name: str = "us-east-1"):
        self.client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.titan-embed-text-v2:0"

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
                embeddings.append([0.0] * 1024) # Fallback dimension for Titan v2
        return embeddings

class RAGManager:
    """Manages the Vector Knowledge Base (ChromaDB) and Retrieval."""
    def __init__(self, session: boto3.Session, region_name: str = "us-east-1"):
        self.embedding_fn = BedrockEmbeddingFunction(session, region_name)
        # Persistent storage in 'chroma_db' folder
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="voice_rag_knowledge",
            embedding_function=self.embedding_fn
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
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

    def retrieve(self, query: str, n_results: int = 3) -> str:
        """Search the knowledge base and return combined context."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        documents = results.get("documents", [[]])[0]
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context = "
---
".join(documents)
        logger.info(f"Retrieved context for query: {query}")
        return context

# --- Strands Tool Definition ---

def create_retrieval_tool(rag_manager: RAGManager):
    """Factory to create a Strands-compatible tool for the BidiAgent."""
    async def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for company policies, documents, or technical info."""
        # Note: BidiAgent tools are async, but ChromaDB/Bedrock calls are sync here.
        # We'll run them in a thread pool if needed, but for prototype, direct call is fine.
        logger.info(f"Agent tool called with query: {query}")
        return rag_manager.retrieve(query)

    # Attach metadata for Strands
    search_knowledge_base.__name__ = "search_knowledge_base"
    search_knowledge_base.__doc__ = "Search the internal knowledge base for specific information, documents, or policies."
    
    return search_knowledge_base
