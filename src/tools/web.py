import logging
import asyncio
import boto3
from botocore.config import Config
from ddgs import DDGS
from strands import tool
from src.core.config import settings
from src.core.prompts import get_web_synthesis_prompt

logger = logging.getLogger(__name__)

def get_web_search_tool(session: boto3.Session):
    bedrock = session.client(
        "bedrock-runtime",
        region_name=settings.AWS_REGION,
        config=Config(read_timeout=3600),
    )

    def _extract_grounding_sources(converse_response: dict, *, limit: int) -> list[str]:
        content_list = (
            converse_response.get("output", {})
            .get("message", {})
            .get("content", [])
        )

        urls: list[str] = []
        for item in content_list:
            citations_content = item.get("citationsContent")
            if not citations_content:
                continue
            citations = citations_content.get("citations") if isinstance(citations_content, dict) else None
            if not isinstance(citations, list):
                continue
            for citation in citations:
                try:
                    url = citation["location"]["web"]["url"]
                except Exception:
                    continue
                if isinstance(url, str) and url and url not in urls:
                    urls.append(url)
                if len(urls) >= limit:
                    return urls
        return urls

    def _urls_to_domains(urls: list[str]) -> list[str]:
        import urllib.parse

        domains: list[str] = []
        for u in urls:
            try:
                host = urllib.parse.urlparse(u).netloc
            except Exception:
                host = ""
            if not host:
                continue
            if host.startswith("www."):
                host = host[4:]
            if host and host not in domains:
                domains.append(host)
        return domains

    @tool(name="web_search", description="Search the internet for real-time news and general knowledge.")
    async def web_search(query: str) -> str:
        max_sources = max(0, min(10, int(getattr(settings, "WEB_SEARCH_MAX_SOURCES", 3))))

        async def try_grounding() -> str | None:
            tool_config = {"tools": [{"systemTool": {"name": "nova_grounding"}}]}
            prompt = (
                "Answer the question using up-to-date web information (web grounding). "
                "Be extremely concise (1-2 sentences). "
                "If you are not confident, say so.\n\n"
                f"QUESTION: {query}"
            )

            model_id = settings.NOVA_GROUNDING_MODEL_ID or settings.NOVA_LITE_MODEL_ID
            try:
                response = await asyncio.to_thread(
                    bedrock.converse,
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    toolConfig=tool_config,
                    inferenceConfig={"maxTokens": 512, "temperature": 0},
                )
            except Exception as e:
                logger.info(f"Nova Grounding (Web) failed: {e}")
                return None

            # Extract text (some SDKs interleave citations + text entries)
            content_list = response.get("output", {}).get("message", {}).get("content", [])
            text_parts: list[str] = []
            for item in content_list:
                t = item.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
            answer = "".join(text_parts).strip()
            if not answer:
                return None

            # Keep citations voice-friendly: include domains only (URLs in logs).
            urls = _extract_grounding_sources(response, limit=max_sources if max_sources > 0 else 0)
            if urls and max_sources > 0:
                domains = _urls_to_domains(urls)[:max_sources]
                if domains:
                    logger.info(f"Nova Grounding (Web) sources: {urls}")
                    answer = f"{answer}\n\nSources: {', '.join(domains)}"

            logger.info(f"Nova Grounding (Web): {answer}")
            return answer

        def ddg_sync():
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=8)

        try:
            backend = (settings.WEB_SEARCH_BACKEND or "auto").lower()
            if backend in {"auto", "grounding"}:
                grounded = await try_grounding()
                if grounded is not None:
                    return grounded
                if backend == "grounding":
                    return "Web search is temporarily unavailable."

            raw_results = await asyncio.to_thread(ddg_sync)
            
            snippets = []
            for r in raw_results:
                snippets.append(f"Title: {r['title']}\nSnippet: {r['body']}")
            
            # Combine raw results and sanitize for Nova Lite
            combined_results = "\n\n".join(snippets)
            # More robust cleaning: remove all non-ASCII printable chars
            clean_text = "".join(c for c in combined_results if c.isprintable() and ord(c) < 128)
            context = clean_text[:2000] # Truncate after cleaning
            
            prompt = get_web_synthesis_prompt(context, query)
            
            response = await asyncio.to_thread(
                bedrock.converse,
                modelId=settings.NOVA_LITE_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 1024, "temperature": 0},
            )
            answer = response['output']['message']['content'][0]['text']
            logger.info(f"Nova Lite (Web): {answer}")
            return answer
        except Exception:
            return "Web search failed."
            
    return web_search
