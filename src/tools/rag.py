import logging
import boto3
from strands import tool
from src.services.knowledge_base import KnowledgeBaseService
from src.core.config import settings
from src.core.prompts import get_rag_synthesis_prompt

logger = logging.getLogger(__name__)

def get_rag_tool(kb: KnowledgeBaseService, session: boto3.Session):
    bedrock = session.client("bedrock-runtime", region_name=settings.AWS_REGION)

    @tool(name="search_documents", description="MANDATORY tool to use when the user asks about uploaded files, PDFs, 'this document', or any specific info. You DO have access to files through this tool.")
    async def search_documents(query: str) -> str:
        context = kb.retrieve(query)
        if "No relevant information" in context: return context

        prompt = get_rag_synthesis_prompt(context, query)
        
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
