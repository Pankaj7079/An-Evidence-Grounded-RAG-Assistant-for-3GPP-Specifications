"""Evidence-grounded text generation with multi-model fallback."""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SYSTEM_PROMPT enforces strict source-tracing — every claim must be anchored to a numbered source block
SYSTEM_PROMPT = """You are a senior 3GPP systems architect. Your answers must have near-zero hallucinations.
You are given numbered source excerpts [S1], [S2], [S3], [S4] from official 3GPP specifications.
Answer using ONLY information that appears in those source excerpts.

FAITHFULNESS GATE (apply before writing every sentence):
Before including any fact, ask yourself: "Which source block [S1-S4] explicitly says this?"
If you cannot point to a specific source block, DO NOT include that fact. Leave it out entirely.
Do not use general 3GPP background knowledge to fill gaps. If the sources don't cover something, acknowledge it briefly.

CITATION RULE:
Inline-cite the clause and page, not the source number: [TS 23.501 Clause 6.2.1, Page 423-424].
Cite once per topic group, not once per sentence.

WRITING RULES:
- Plain engineering English. Synthesize and explain — never paste raw spec text verbatim.
- No filler openers: "Certainly", "Of course", "It is important to note", "The X is a critical component".
- Do not use dollar signs or section symbols (§).
- End your answer cleanly. No trailing lists of terms after your final paragraph.
- Vary sentence structure. Avoid repeating the same opening pattern across paragraphs.

STRUCTURE (choose based on the question type):

1. CONCEPTUAL ("What is X?", "What does Y do?"):
   2-3 paragraphs. Explain purpose, key behaviors, and context in the architecture.
   Add a bullet list only if 5+ genuinely separate items are cleaner as a list than prose.
   Cite once per paragraph.

2. PROCEDURAL / FLOW ("Explain the X procedure", "X flow", "How does X work?"):
   One intro sentence — what is the purpose of this procedure and what triggers it — with citation.
   Then numbered steps derived ONLY from the source excerpts.
   If a source only covers part of the flow, describe that part and clearly note the sources are partial.
   Do NOT add steps from memory or general knowledge.

3. COMPONENT / FUNCTION LIST ("What are the functions of X?", "Core functions of X"):
   Intro paragraph (3-4 sentences): what the component is, where it sits in 5G Core, why it matters.
   Then 2-3 H4 headings grouping related functions logically.
   Each heading ends with a citation: #### NAS & Registration Management [TS 23.501 Clause 6.2.1, Page 423]
   Under each heading: 2-3 concise bullets explaining in plain language what these functions do.

4. OVERVIEW / BROAD ("Tell me about 5G", "How does 5G work?"):
   Two focused paragraphs — one on components/architecture, one on design principles or capabilities.
   Be selective. Cover what the sources actually say, not a generic 5G overview from memory.
   Cite once per paragraph.

NEGATIVE QUERIES:
If the sources don't contain enough information to answer, respond with exactly:
"I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."
"""


def clean_output_formatting(text: str) -> str:
    """Clean up formatting artifacts and escape characters that break Streamlit rendering."""
    lower = text.lower().strip()

    # Normalize out-of-scope / negative responses to canonical abstention sentence
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

    # Escape dollar signs to prevent Streamlit LaTeX rendering
    text = text.replace("$", "\\$")
    # Replace section signs with plain text
    text = text.replace("§", "Clause ")

    # Strip trailing orphaned bullet lines (e.g. "• N4 Session Establishment • ...") after final paragraph
    lines = text.split("\n")
    # Remove trailing lines that are pure bullet artifacts with no cited content
    while lines and re.match(r"^[•·▪\-\*]\s", lines[-1].strip()) and "[TS" not in lines[-1]:
        lines.pop()
    text = "\n".join(lines)

    # Normalize excessive whitespace artifacts
    text = re.sub(r"\s*``\s*", " ", text)
    text = re.sub(r"\s*''\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove 3+ consecutive blank lines down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
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
        temperature: float = 0.05,
        max_tokens: int = settings.MAX_OUTPUT_TOKENS,
    ) -> str:
        """Generate response with automatic retry and model fallback."""
        # temperature=0.05: near-deterministic for faithfulness, slight variation avoids mechanical repetition
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
    """Format grounded prompt with explicitly labeled source blocks and adaptive style guidance."""
    context_blocks = [
        "=== GROUNDING SOURCES (use ONLY these to answer) ===",
        "Every fact in your answer must come from one of these numbered source blocks.",
    ]

    for idx, chunk in enumerate(retrieved_chunks, 1):
        clean_chunk_text = chunk.text.strip()
        # Trim long chunks to stay within token budget while keeping the most relevant content
        if len(clean_chunk_text) > 1000:
            clean_chunk_text = clean_chunk_text[:1000] + "..."

        # Label each source block as [S1], [S2] etc — LLM must trace claims to these labels
        context_blocks.append(
            f"[S{idx}] {chunk.document_code} | Clause {chunk.section_number}: {chunk.section_title} | Page {chunk.page_number}\n"
            f"{clean_chunk_text}\n"
        )

    context_str = "\n".join(context_blocks)

    # Detect question type for adaptive structure guidance
    q_type = _detect_question_type(query)

    if q_type == "procedural":
        style_hint = (
            "QUESTION TYPE: PROCEDURAL / FLOW.\n"
            "Write: (1) One intro sentence on what this procedure does and what triggers it, with inline clause citation. "
            "(2) Numbered steps — ONLY steps you can directly trace to [S1]-[S4] above. "
            "If a source covers only part of the flow, describe that part and note the sources are partial. "
            "Do NOT add steps from memory or general 3GPP knowledge. Total: 200-280 words."
        )
    elif q_type == "comparison":
        style_hint = (
            "QUESTION TYPE: COMPONENT / FUNCTION LIST.\n"
            "Write a 3-4 sentence intro paragraph on this component's role in the 5G Core. "
            "Then 2-3 H4 headings grouping related functions, each ending with its clause citation. "
            "Under each heading: 2-3 bullets in plain language — only include what [S1]-[S4] explicitly state. "
            "Total: 220-290 words."
        )
    elif q_type == "overview":
        style_hint = (
            "QUESTION TYPE: BROAD OVERVIEW.\n"
            "Write 2 focused prose paragraphs — one on components/how they connect, one on key design principles. "
            "Use only facts explicitly stated in [S1]-[S4]. Cite once per paragraph. Total: 160-240 words."
        )
    else:  # conceptual
        style_hint = (
            "QUESTION TYPE: CONCEPTUAL.\n"
            "Write 2-3 paragraphs: purpose of this feature, how it works, where it fits in 5G architecture. "
            "Use only facts from [S1]-[S4]. Conversational but precise. Cite once per paragraph. Total: 180-260 words."
        )

    user_prompt = (
        f"{context_str}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"{style_hint}\n"
        f"REMINDER: Before each sentence, verify it traces to [S1], [S2], [S3], or [S4] above. "
        f"If it doesn't, cut it."
    )
    return user_prompt
