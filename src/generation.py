"""Evidence-Grounded Generation Engine with Dynamic Technical Structuring.

Supports high-throughput Groq models (Llama-3.1-8b, Llama-3-70b, Qwen-2.5, DeepSeek)
and Google Gemini with automatic rate-limit resilience.
"""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior 3GPP systems architect. Answer questions using ONLY the provided specification excerpts.

STRUCTURE:
1. Start with a direct opening paragraph (2-3 sentences) that answers the question. End this paragraph with the primary citation in brackets, e.g. [TS 23.501 Clause 6.2.1, Page 423-424].
2. Add 2-3 short sections using Markdown H4 headings (####). Each heading MUST end with its citation tag in brackets, e.g.:
   #### Access and Mobility Control [TS 23.501 Clause 6.2.1, Page 423-424]
3. Under each heading, write 2-4 concise bullet points using hyphens (-). Group related items into a single bullet. Do NOT repeat citations on individual bullets.
4. Keep total answer to 200-300 words.

WRITING RULES:
- Write naturally and vary sentence structure.
- Synthesize and explain; don't copy-paste raw spec text.
- No filler phrases: "It is important to note", "As specified in", "According to the context".
- Do not use dollar signs in your output.
- Every claim must trace back to the provided excerpts.

NEGATIVE QUERIES:
If the excerpts don't cover the question, respond with exactly:
"I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."
No headers, bullets, or citations for negative responses.
"""


def clean_output_formatting(text: str) -> str:
    """Clean up formatting and escape characters that break Streamlit rendering."""
    # Abstention: return just the abstention sentence
    if "could not find sufficient supporting evidence" in text.lower():
        for line in text.split("\n"):
            if "could not find sufficient supporting evidence" in line.lower():
                return line.strip()

    # Escape dollar signs — Streamlit renders them as LaTeX math delimiters
    text = text.replace("$", "\\$")

    # Remove empty backticks or double spaces
    text = re.sub(r"\s*``\s*", " ", text)
    text = re.sub(r"\s*''\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class LLMClient:
    """Unified LLM client interface with multi-model fallback."""

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

        # Modern active Groq models in priority order
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ]

        # Initialize provider client
        self._groq_client = None
        self._gemini_client = None

        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, max_retries=0)
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
        """Generate deterministic response with seamless multi-model fallback."""
        # 1. Try Groq models in sequence
        if self._groq_client:
            for model_name in self.groq_models:
                try:
                    res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model=model_name)
                    return clean_output_formatting(res)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "rate limit" in err_msg:
                        logger.warning(f"Groq {model_name} rate limit, brief pause...")
                        time.sleep(2.0)
                    else:
                        logger.warning(f"Groq {model_name} failed: {e}")

        # 2. Try Gemini
        if self._gemini_client:
            try:
                res = self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
                return clean_output_formatting(res)
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}.")

        # 3. Final single attempt on primary groq model after backoff
        if self._groq_client:
            time.sleep(3.0)
            res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model="llama-3.1-8b-instant")
            return clean_output_formatting(res)

        raise RuntimeError("All LLM providers and fallback models are currently unavailable.")

    def _generate_groq(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        target_model = model or "llama-3.1-8b-instant"
        response = self._groq_client.chat.completions.create(
            model=target_model,
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
    """Format prompt with compact 3GPP clause excerpts to stay within rate limits."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        # Truncate chunk text to ~800 chars to maintain compact prompt tokens
        clean_chunk_text = chunk.text.strip()
        if len(clean_chunk_text) > 850:
            clean_chunk_text = clean_chunk_text[:850] + "..."

        context_blocks.append(
            f"--- Context Source [{idx}] ---\n"
            f"Specification: {chunk.document_code} (Rel-{chunk.release})\n"
            f"Clause: {chunk.section_number} - {chunk.section_title}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Content:\n{clean_chunk_text}\n"
        )

    context_str = "\n".join(context_blocks)
    user_prompt = (
        f"{context_str}\n\n"
        f"=== USER QUESTION ===\n{query}\n\n"
        f"Answer concisely in 150-250 words. Synthesize — don't dump raw bullet lists. "
        f"Cite each clause once (e.g. [TS 23.501 Clause X.Y, Page Z]) and group related points together."
    )
    return user_prompt
