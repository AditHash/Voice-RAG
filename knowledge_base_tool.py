import logging
from strands import tool
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

def get_knowledge_base_tool(kb: KnowledgeBase):
    """Factory to create the RAG search tool for the BidiAgent."""
    
    @tool(name="search_documents", description="MANDATORY tool to use when the user asks about uploaded files, PDFs, 'this document', or any specific info that might be in a document.")
    async def search_documents(query: str) -> str:
        logger.info(f"Tool: Searching documents for '{query}'")
        return kb.retrieve(query)
        
    return search_documents
