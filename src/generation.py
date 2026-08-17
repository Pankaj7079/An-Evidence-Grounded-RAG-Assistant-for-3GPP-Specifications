"""Evidence-grounded text generation with multi-model fallback."""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Adaptive system prompt — structure follows the nature of the question, not a rigid template
SYSTEM_PROMPT = """You are a senior 3GPP systems architect with 15 years of experience. Answer questions using ONLY the provided specification excerpts.

CORE RULES:
- Cite every claim inline like [TS 23.501 Clause 6.2.1, Page 423-424]. One citation per topic, not per sentence.
- Do not use dollar signs. Do not use section signs (§).
- Synthesize the spec text — never paste raw text verbatim.
- No filler openers: never start with "Certainly", "Of course", "As per the spec", "It is important to note", "According to the context".
- Every fact must trace back to the provided excerpts.

STRUCTURE — adapt to the question type:

1. CONCEPTUAL questions ("What is X?", "What does X do?"): 
   Write 2-3 natural explanatory paragraphs. Keep it conversational but precise. Inline-cite once per paragraph.
   Only add a short bulleted list at the end if there are genuinely 4+ distinct items worth listing.

2. PROCEDURAL / FLOW questions ("How does X work?", "Explain the X procedure", "X flow"):
   Brief intro paragraph (2 sentences, 1 citation), then numbered steps or a short sequenced list.
   Group related steps — don't fragment every sentence into its own bullet.

3. COMPARISON / MULTI-PART questions ("Difference between X and Y", "What are the types of X?"):
   Short intro sentence. Then use 2-3 H4 headings (####), each with 2-3 tight bullets.
   Each H4 heading must end with its citation, e.g.: #### AMF Functions [TS 23.501 Clause 6.2.1, Page 423-424]

4. BROAD / OVERVIEW questions ("Tell me about 5G", "Explain the 5GS architecture"):
   2 focused paragraphs covering the key points from the excerpts. Don't try to cover everything — pick the most important ideas and explain them clearly.

NEGATIVE QUERIES:
If the excerpts don't contain enough information to answer, respond with exactly this sentence and nothing else:
"I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."
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


def _detect_question_type(query: str) -> str:
    """Classify query intent to guide answer structure."""
    q = query.lower().strip()

    # Procedural / flow patterns
    if any(kw in q for kw in ["procedure", "flow", "how does", "how do", "explain the", "establishment", "registration", "handover", "sequence", "steps", "process"]):
        return "procedural"

    # Comparison / multi-part patterns
    if any(kw in q for kw in ["difference", "compare", "vs", "types of", "list", "what are the", "core functions", "main functions", "key features", "roles"]):
        return "comparison"

    # Broad / overview patterns
    if any(kw in q for kw in ["tell me about", "overview", "explain 5g", "how 5g", "what is 5g", "network working", "architecture"]):
        return "overview"

    # Default: conceptual
    return "conceptual"


def format_grounded_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
) -> str:
    """Format user prompt with context excerpts and adaptive structure guidance."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        clean_chunk_text = chunk.text.strip()
        # Keep chunk text compact to stay within token budgets
        if len(clean_chunk_text) > 900:
            clean_chunk_text = clean_chunk_text[:900] + "..."

        context_blocks.append(
            f"--- Context Source [{idx}] ---\n"
            f"Specification: {chunk.document_code} (Rel-{chunk.release})\n"
            f"Clause: {chunk.section_number} - {chunk.section_title}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Content:\n{clean_chunk_text}\n"
        )

    context_str = "\n".join(context_blocks)

    # Choose structure guidance based on detected question type
    q_type = _detect_question_type(query)

    if q_type == "procedural":
        style_hint = (
            "This is a PROCEDURAL question. Write a short intro (1-2 sentences, 1 citation), "
            "then describe the flow using numbered steps or a tight sequenced list. "
            "Keep total answer to 180-250 words. Group related steps — don't fragment each sentence into its own bullet."
        )
    elif q_type == "comparison":
        style_hint = (
            "This is a COMPARISON/LIST question. Write a single intro sentence, then use 2-3 H4 headings (####) "
            "each ending with its citation e.g. #### AMF Functions [TS 23.501 Clause 6.2.1, Page 423]. "
            "Under each heading write 2-3 tight bullet points. Total answer: 200-270 words."
        )
    elif q_type == "overview":
        style_hint = (
            "This is a BROAD overview question. Write 2 focused explanatory paragraphs covering the most "
            "important ideas from the excerpts. Don't try to cover everything — be selective and clear. "
            "Inline-cite once per paragraph. Total answer: 150-230 words."
        )
    else:  # conceptual
        style_hint = (
            "This is a CONCEPTUAL question. Write 2-3 natural paragraphs that explain the concept clearly. "
            "Inline-cite once per paragraph. Only add a brief bullet list if there are 4+ genuinely distinct items. "
            "Total answer: 170-250 words."
        )

    user_prompt = (
        f"{context_str}\n\n"
        f"=== USER QUESTION ===\n{query}\n\n"
        f"{style_hint}"
    )
    return user_prompt
