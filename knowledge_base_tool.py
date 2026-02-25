import logging
from strands import tool
from knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

def get_knowledge_base_tool(kb: KnowledgeBase):
    """Factory to create the RAG search tool for the BidiAgent."""
    
    @tool(description="Search the internal knowledge base for specific information, company documents, or technical info.")
    async def search_knowledge_base(query: str) -> str:
        logger.info(f"Tool: Searching local knowledge base for '{query}'")
        return kb.retrieve(query)
        
    return search_knowledge_base
