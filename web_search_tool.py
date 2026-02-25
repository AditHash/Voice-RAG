import logging
import asyncio
import boto3
from ddgs import DDGS
from strands import tool
from config import Config

logger = logging.getLogger(__name__)

def get_web_search_tool(session: boto3.Session):
    """Factory to create the web search tool with Nova Lite synthesis."""
    
    bedrock = session.client("bedrock-runtime", region_name=Config.AWS_REGION)

    @tool(name="web_search_tool", description="Perform a deep web search to find real-time news, current events, or general info. This tool searches the internet and synthesizes a professional answer.")
    async def web_search_tool(query: str) -> str:
        """Perform a comprehensive web search using DuckDuckGo and synthesize with Nova Lite."""
        logger.info(f"Tool: Deep searching and synthesizing for: {query}")
        
        def run_search():
            with DDGS() as ddgs:
                return ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=8)

        try:
            search_results = await asyncio.to_thread(run_search)
            
            snippets = []
            for r in search_results:
                snippets.append(f"Title: {r['title']}\nSnippet: {r['body']}")
            
            if not snippets:
                return f"No detailed results found for '{query}' on the web."
            
            # Synthesize results using Nova Lite
            context = "\n---\n".join(snippets)
            prompt = f"""You are a professional researcher. Based on the following WEB SEARCH RESULTS, answer the USER QUERY.
            
            WEB RESULTS:
            {context}
            
            USER QUERY:
            {query}
            
            INSTRUCTIONS:
            - Be extremely concise (max 2 sentences).
            - Focus on facts and real-time info.
            - Speak naturally for a voice assistant.
            """

            logger.info(f"Synthesis: Sending web results to Nova Lite...")
            response = bedrock.converse(
                modelId=Config.NOVA_LITE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 200, "temperature": 0}
            )
            answer = response['output']['message']['content'][0]['text']
            logger.info(f"Nova Lite: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Web search or synthesis failed: {e}")
            return "The web search service is currently unavailable."
            
    return web_search_tool
