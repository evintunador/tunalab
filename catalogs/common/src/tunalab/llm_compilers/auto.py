"""
Auto-detect and return an appropriate LLMClient based on the current environment.

Priority order:
  1. AWS Bedrock  — when CLAUDE_CODE_USE_BEDROCK=1 (or AWS_ACCESS_KEY_ID is set)
  2. Anthropic    — when ANTHROPIC_API_KEY is set
  3. None         — caller decides what to do (smart_train falls back to mock)
"""

import os
from typing import Optional

from tunalab.protocols.llm_client import LLMClient


def get_default_llm_client() -> Optional[LLMClient]:
    """Return a ready-to-use LLMClient for the current environment, or None.

    Checks environment variables in priority order:
    - AWS Bedrock if ``CLAUDE_CODE_USE_BEDROCK`` is set to a truthy value,
      or if ``AWS_ACCESS_KEY_ID`` is present (indicating Bedrock credentials
      have been configured via the standard AWS env vars).
    - Anthropic direct API if ``ANTHROPIC_API_KEY`` is set.
    - Returns ``None`` if neither is available.
    """
    use_bedrock = os.getenv("CLAUDE_CODE_USE_BEDROCK", "").strip() in ("1", "true", "yes")
    has_aws_creds = bool(os.getenv("AWS_ACCESS_KEY_ID"))

    if use_bedrock or has_aws_creds:
        try:
            from tunalab.llm_compilers.bedrock import BedrockLLM
            return BedrockLLM()
        except Exception:
            pass  # fall through to Anthropic

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from tunalab.llm_compilers.anthropic import AnthropicLLM
            return AnthropicLLM(api_key=anthropic_key)
        except Exception:
            pass

    return None
