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


class OpenAILLM:
    """
    OpenAI LLM client using litellm for completion.
    Accepts 'model' like 'gpt-4o' or 'openai/gpt-4o'.
    
    Implements the LLMClient protocol from tunalab.protocols.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.model = model if "/" in model else f"openai/{model}"

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
                prior_code=prior_code
            )
        )