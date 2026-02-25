import logging
import boto3
from strands import tool
from src.voice_rag.services.knowledge_base import KnowledgeBaseService
from src.voice_rag.core.config import settings

logger = logging.getLogger(__name__)

def get_rag_tool(kb: KnowledgeBaseService, session: boto3.Session):
    bedrock = session.client("bedrock-runtime", region_name=settings.AWS_REGION)

    @tool(name="search_documents", description="Use to search uploaded files and PDFs for specific info.")
    async def search_documents(query: str) -> str:
        context = kb.retrieve(query)
        if "No relevant information" in context: return context

        prompt = f"Using this context: {context}

Answer concisely: {query}"
        
        try:
            response = bedrock.converse(
                modelId=settings.NOVA_LITE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0}
            )
            answer = response['output']['message']['content'][0]['text']
            logger.info(f"Nova Lite (RAG): {answer}")
            return answer
        except Exception:
            return context[:500]
            
    return search_documents
