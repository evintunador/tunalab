"""Protocol for LLM providers used in code compilation."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Interface for LLM providers used in code compilation."""
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate code from system and user prompts."""
        ...
    
    def refine(self, system_prompt: str, user_prompt: str, prior_code: str, error_summary: str) -> str:
        """Refine prior code based on error feedback."""
        ...


def strip_code_fences(text: str) -> str:
    """Helper to strip markdown code fences from LLM output."""
    if text is None:
        return ""
    # Remove triple backtick fences if present
    if "```" in text:
        # take inside of first/last fence as best-effort
        parts = text.split("```")
        # odd indices are fenced blocks; prefer first fenced block
        for i in range(1, len(parts), 2):
            if parts[i].strip():
                return parts[i].lstrip("python").lstrip()
        # fallback to concat of all non-fence segments
        return "".join(p for i, p in enumerate(parts) if i % 2 == 0)
    return text

