import logging
import json
import boto3
from strands import tool
from knowledge_base import KnowledgeBase
from config import Config

logger = logging.getLogger(__name__)

def get_knowledge_base_tool(kb: KnowledgeBase, session: boto3.Session):
    """Factory to create the RAG search tool with Nova Lite synthesis."""
    
    # Create a Bedrock client for Nova Lite synthesis
    bedrock = session.client("bedrock-runtime", region_name=Config.AWS_REGION)

    @tool(name="search_documents", description="MANDATORY tool to use when the user asks about uploaded files, PDFs, or specific info. This tool searches the knowledge base and synthesizes an answer.")
    async def search_documents(query: str) -> str:
        logger.info(f"Tool: Searching and synthesizing answer for '{query}'")
        
        # 1. Retrieve raw chunks
        context = kb.retrieve(query)
        if "No relevant information found" in context:
            return context

        # 2. Synthesize using Nova Lite
        # This makes the bot "smarter" than raw Sonic synthesis
        prompt = f"""You are a professional assistant. Based on the provided DOCUMENT CONTEXT, answer the USER QUERY.
        
        DOCUMENT CONTEXT:
        {context}
        
        USER QUERY:
        {query}
        
        INSTRUCTIONS:
        - Be extremely concise (1-2 sentences).
        - Use a natural, conversational tone for voice.
        - Only use info from the context. If not found, say you don't know.
        """

        try:
            logger.info(f"Synthesis: Sending context to Nova Lite...")
            response = bedrock.converse(
                modelId=Config.NOVA_LITE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0}
            )
            answer = response['output']['message']['content'][0]['text']
            logger.info(f"Nova Lite: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Nova Lite synthesis failed: {e}")
            return f"Raw Data from Documents: {context[:500]}..."
        
    return search_documents
