"""Evidence-Grounded Generation Engine with Strict Citation Enforcement.

Supports Groq (Llama-3.3-70b-versatile) and Google Gemini with temperature=0.0
for deterministic, hallucination-free generation grounded exclusively in 3GPP specifications.
"""

import logging
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Principal 3GPP Telecom System Architect assisting engineering teams with 3GPP 5G Specifications (TS 23.501 & TS 23.502).

PRIMARY DIRECTIVE:
You must answer questions strictly and exclusively using the provided 3GPP specification excerpts.

ANSWER STRUCTURE & CITATION RULES:
1. Provide a clean, executive-level technical answer formatted with clear Markdown sections:
   - `### Executive Summary`: A crisp 1-2 sentence direct response to the query.
   - `### Functional Capabilities / Key Procedures`: Categorized bullet points grouped logically by functional domain.
   - `### Applicable Reference Points & Interfaces`: Summary of relevant reference points (e.g. N1, N2, N4), if applicable.
2. CITATION PLACEMENT RULE:
   - Place exact inline citations naturally at the end of each major concept, category, or bullet point: `[TS 23.501 Clause X.Y, Page Z]` or `[TS 23.502 Clause X.Y, Page Z]`.
   - DO NOT spam or repeat the exact same citation bracket on every single minor sub-bullet if they all derive from the same clause. Cite the source once per logical section or distinct claim.
   - For multi-page clauses, cite the range: `[TS 23.501 Clause 4.2.4, Page 42-43]`.
   - Only cite clauses and page numbers present in the context source headers.

NEGATIVE & SCOPE CONSTRAINTS:
- If the query asks about a concept or feature not present in the 3GPP context (e.g. 6G, quantum teleportation, non-telecom topics), state directly:
  "I could not find sufficient supporting evidence in the indexed 3GPP documents for this query."
- DO NOT list unrelated 5G procedures when answering negative queries.
- NEVER include robotic meta-commentary, apologies, or disclaimers such as "There is no mention of TS 23.502 in the provided context...", "Based on the provided excerpts...", or "According to the retrieved text...".
"""


class LLMClient:
    """Unified LLM client interface for Groq and Google Gemini."""

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY
        self.groq_model = groq_model or settings.GROQ_MODEL
        self.gemini_model = gemini_model or "gemini-flash-latest"

        # Initialize provider client
        self._groq_client = None
        self._gemini_client = None

        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = settings.MAX_OUTPUT_TOKENS,
    ) -> str:
        """Generate deterministic response from configured LLM provider."""
        if self.provider == "groq" and self._groq_client:
            try:
                return self._generate_groq(prompt, system_instruction, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Groq generation failed: {e}. Attempting fallback to Gemini...")
                if self._gemini_client:
                    return self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
                raise

        if self.provider == "gemini" and self._gemini_client:
            try:
                return self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}. Attempting fallback to Groq...")
                if self._groq_client:
                    return self._generate_groq(prompt, system_instruction, temperature, max_tokens)
                raise

        # Default fallback to Groq if available
        if self._groq_client:
            return self._generate_groq(prompt, system_instruction, temperature, max_tokens)

        raise RuntimeError("No LLM provider client is configured or available.")

    def _generate_groq(
        self, prompt: str, system_instruction: str, temperature: float, max_tokens: int
    ) -> str:
        response = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(
        self, prompt: str, system_instruction: str, temperature: float, max_tokens: int
    ) -> str:
        full_content = f"{system_instruction}\n\nUser Question:\n{prompt}"
        response = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=full_content,
        )
        return response.text.strip()


def format_grounded_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
) -> str:
    """Format prompt with retrieved 3GPP clause excerpts and page numbers."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- Context Source [{idx}] ---\n"
            f"Specification: {chunk.document_code} (Rel-{chunk.release})\n"
            f"Clause: {chunk.section_number} - {chunk.section_title}\n"
            f"Hierarchy: {chunk.section_hierarchy}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )

    context_str = "\n".join(context_blocks)
    user_prompt = (
        f"{context_str}\n\n"
        f"=== USER QUESTION ===\n{query}\n\n"
        f"Provide an evidence-grounded answer based strictly on the context above. "
        f"Every factual claim must cite its source in the format `[TS 23.501 Clause X.Y, Page Z]` or `[TS 23.502 Clause X.Y, Page Z]`."
    )
    return user_prompt
