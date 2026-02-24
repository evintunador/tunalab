from typing import Optional
import os

from litellm import completion
from tunalab.protocols.llm_client import strip_code_fences


REFINE_USER_PROMPT = \
"""{user_prompt}

Fix the prior code to address the following errors. Output ONLY a single complete Python file (no backticks, no commentary).

Errors to fix:
{error_summary}

Prior code:
{prior_code}
"""


class BedrockLLM:
    """
    AWS Bedrock LLM client using litellm for completion.

    Reads credentials from the standard AWS environment variables
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN) and
    region from AWS_REGION or AWS_REGION_NAME.  No API key needed.

    Accepts model IDs in the form used by .bashrc Bedrock aliases, e.g.
    'us.anthropic.claude-sonnet-4-6' or 'global.anthropic.claude-opus-4-6-v1'.
    The 'bedrock/' prefix and any '[1m]' context-size suffix are handled
    automatically so callers don't need to worry about litellm formatting.

    Implements the LLMClient protocol from tunalab.protocols.
    """

    DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"

    def __init__(self, model: str = DEFAULT_MODEL, region: Optional[str] = None):
        # Strip the optional [1m] / [Xm] context-window suffix that Bedrock
        # model IDs sometimes carry — litellm doesn't understand it.
        model_id = model.split("[")[0]

        # Ensure the litellm bedrock/ prefix is present.
        self.model = model_id if model_id.startswith("bedrock/") else f"bedrock/{model_id}"

        # litellm reads AWS_REGION_NAME; also accept the shorter AWS_REGION.
        resolved_region = region or os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION")
        if resolved_region:
            os.environ["AWS_REGION_NAME"] = resolved_region

        # Fail early if credentials are missing.
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            raise ValueError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID / "
                "AWS_SECRET_ACCESS_KEY (and optionally AWS_SESSION_TOKEN) "
                "before using BedrockLLM."
            )

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = completion(model=self.model, messages=messages)
        return strip_code_fences(resp.choices[0].message.content)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._chat(system_prompt, user_prompt)

    def refine(self, system_prompt: str, user_prompt: str, prior_code: str, error_summary: str) -> str:
        return self._chat(
            system_prompt,
            REFINE_USER_PROMPT.format(
                user_prompt=user_prompt,
                error_summary=error_summary,
                prior_code=prior_code,
            ),
        )
