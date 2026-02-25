import os
import json
import logging
import boto3
import chromadb
import fitz
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import EmbeddingFunction
from src.core.config import settings

logger = logging.getLogger(__name__)

class BedrockEmbeddingFunction(EmbeddingFunction):
    def __init__(self, session: boto3.Session):
        self.client = session.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self.model_id = settings.TITAN_EMBED_MODEL_ID

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
                logger.error(f"Embedding error: {e}")
                embeddings.append([0.0] * 1024)
        return embeddings

class KnowledgeBaseService:
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

    def ingest_text(self, text: str, metadata: Dict[str, Any] = None):
        chunks = self.text_splitter.split_text(text)
        ids = [f"chunk_{os.urandom(4).hex()}" for _ in range(len(chunks))]
        metadatas = [metadata or {} for _ in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        return self.ingest_text(full_text, {"filename": filename, "type": "pdf"})

    def clear_all(self):
        try:
            self.chroma_client.delete_collection(name=settings.COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            return True
        except Exception:
            return False

    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents in the knowledge base."""
        try:
            results = self.collection.get(include=["metadatas"])
            unique_files = {}
            for metadata in results.get("metadatas", []):
                filename = metadata.get("filename")
                if filename:
                    unique_files[filename] = unique_files.get(filename, 0) + 1
            
            return [{"filename": fn, "chunks": count} for fn, count in unique_files.items()]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def retrieve(self, query: str, n_results: int = 2) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i in range(len(documents)):
            filename = metadatas[i].get("filename", "Unknown")
            text = documents[i][:800]
            # More robust cleaning: remove all non-ASCII printable chars
            clean_text = "".join(c for c in text if c.isprintable() and ord(c) < 128)
            context_parts.append(f"[Source: {filename}]\n{clean_text}")
            
        return "\n---\n".join(context_parts)
