import logging
import asyncio
import boto3
from ddgs import DDGS
from strands import tool
from src.voice_rag.core.config import settings

logger = logging.getLogger(__name__)

def get_web_search_tool(session: boto3.Session):
    bedrock = session.client("bedrock-runtime", region_name=settings.AWS_REGION)

    @tool(name="web_search", description="Search the internet for real-time news and general knowledge.")
    async def web_search(query: str) -> str:
        def ddg_sync():
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=8)

        try:
            raw_results = await asyncio.to_thread(ddg_sync)
            context = "
".join([f"{r['title']}: {r['body']}" for r in raw_results])
            
            prompt = f"Summarize these web results concisely for voice: {context}

Query: {query}"
            
            response = bedrock.converse(
                modelId=settings.NOVA_LITE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0}
            )
            answer = response['output']['message']['content'][0]['text']
            logger.info(f"Nova Lite (Web): {answer}")
            return answer
        except Exception:
            return "Web search failed."
            
    return web_search
