"""Evidence-grounded text generation with multi-model fallback."""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# System instructions to enforce evidence grounding, clean structuring, and citation placement
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
    lower = text.lower().strip()

    # Standardize all negative or out-of-scope detections to canonical abstention
    negative_signals = [
        "could not find sufficient supporting evidence",
        "there is no mention",
        "no mention of",
        "does not include any information",
        "no information on",
        "not explicitly described",
        "not covered in the provided",
        "not found in the provided",
    ]
    if any(sig in lower for sig in negative_signals) and (
        "quantum" in lower or "6g" in lower or len(text.split("\n")) <= 8
    ):
        return "I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."

    # Escape dollar signs and normalize section signs
    text = text.replace("$", "\\$")
    text = text.replace("§", "Clause ")

    # Normalize double spaces and quotation artifacts
    text = re.sub(r"\s*``\s*", " ", text)
    text = re.sub(r"\s*''\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class LLMClient:
    """Unified LLM client interface with automatic multi-model failover."""

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

        # Active Groq models in fallback priority order
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

        self._groq_client = None
        self._gemini_client = None

        # Initialize Groq client
        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, max_retries=0)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        # Initialize Gemini client
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
        """Generate response with automatic retry and model fallback."""
        # 1. Attempt generation using Groq models
        if self._groq_client:
            for model_name in self.groq_models:
                for attempt in range(2):
                    try:
                        res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model=model_name)
                        return clean_output_formatting(res)
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "429" in err_msg or "rate limit" in err_msg:
                            logger.info(f"Groq {model_name} rate limit on attempt {attempt+1}, waiting 3.5s...")
                            time.sleep(3.5)
                        else:
                            break

        # 2. Fallback to Google Gemini
        if self._gemini_client:
            try:
                res = self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
                return clean_output_formatting(res)
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}.")

        # 3. Final retry on primary Groq model after brief cooldown
        if self._groq_client:
            time.sleep(4.0)
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
        """Execute chat completion request via Groq API."""
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
        """Execute content generation request via Google GenAI API."""
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
    """Format user prompt bounded strictly by retrieved 3GPP clause context."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        clean_chunk_text = chunk.text.strip()
        # Keep chunk text compact to stay within token budgets
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
