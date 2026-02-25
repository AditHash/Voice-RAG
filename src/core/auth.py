import os
import boto3
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)

def get_aws_session() -> boto3.Session:
    """Universal Auth Resolver: Returns a signed AWS session based on available credentials."""
    
    if settings.BEDROCK_API_KEY:
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = settings.BEDROCK_API_KEY
        logger.info("Auth: Configured Bedrock Bearer Token.")

    # Prioritize SigV4 keys if present
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        os.environ.pop("AWS_PROFILE", None)
        os.environ.pop("AWS_DEFAULT_PROFILE", None)
        logger.info("Auth: Using SigV4 keys from environment.")

    return boto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        aws_session_token=settings.AWS_SESSION_TOKEN,
        region_name=settings.AWS_REGION
    )
