import anthropic, os, json
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask(prompt: str, system: str = "", temp: float = 0.3) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, temperature=temp,
        system=system, messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip()


# ══════════════════════════════════════════════════════
# WHAT IS HALLUCINATION?
# The model generates plausible-sounding but false information
# Happens because LLMs predict "likely next tokens", not facts
# ══════════════════════════════════════════════════════

# ── Example 1: Knowledge cutoff hallucination ─────────
print("=== Hallucination Example (knowledge cutoff) ===")
risky = ask("What is the current price of Anthropic's API per token?", temp=0.7)
print(f"Risky (no grounding): {risky}")
# May give outdated or wrong prices — model's training data is stale


# ── Example 2: Fabricated citations ───────────────────
print("\n=== Fabricated Citation Risk ===")
citation_risk = ask(
    "Cite 3 academic papers about RAG (Retrieval-Augmented Generation) with authors and years.",
    temp=0.5
)
print(citation_risk)
# WARNING: Some of these paper titles/authors may be invented!
# Never trust LLM-generated citations without verification


# ══════════════════════════════════════════════════════
# FIX 1: GROUNDING — Give Claude the facts to work from
# This is the #1 technique FDEs use (foundation of RAG)
# ══════════════════════════════════════════════════════
print("\n=== Fix 1: Grounding with context ===")

# Simulate retrieved knowledge base content
KNOWLEDGE_BASE = """
[Source: Anthropic Pricing Page, retrieved 2025-04]
Claude Sonnet 4.6 pricing:
- Input:  $3.00 per million tokens
- Output: $15.00 per million tokens
- Context window: 200,000 tokens
- Free tier: Not available for production
- Enterprise: Custom pricing with volume discounts
"""

grounded_prompt = f"""Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

<context>
{KNOWLEDGE_BASE}
</context>

<question>What is the price per million tokens for Claude Sonnet 4.6 output?</question>"""

result = ask(grounded_prompt, temp=0.0)
print(result)
# Now it answers correctly from the grounded source


# ══════════════════════════════════════════════════════
# FIX 2: INSTRUCTION TO ADMIT UNCERTAINTY
# Tell Claude explicitly to say "I don't know"
# ══════════════════════════════════════════════════════
print("\n=== Fix 2: Uncertainty admission ===")

honest_system = """You are a precise technical assistant.
Rules:
- If you are not 100% certain of a fact, say "I'm not certain, but..."
- If you don't know something, say "I don't have reliable information on this."
- Never make up statistics, dates, or names.
- Cite your confidence level when relevant (high/medium/low)."""

result = ask(
    "What is the exact token limit for GPT-4o as of this month?",
    system=honest_system,
    temp=0.0
)
print(result)
# Should express uncertainty rather than giving a potentially wrong number


# ══════════════════════════════════════════════════════
# FIX 3: SELF-CONSISTENCY CHECK
# Ask Claude to verify its own answer
# ══════════════════════════════════════════════════════
print("\n=== Fix 3: Self-consistency check ===")

def verified_answer(question: str, context: str = "") -> dict:
    # Step 1: Get initial answer
    ctx_block = f"<context>{context}</context>\n\n" if context else ""
    initial = ask(f"{ctx_block}<question>{question}</question>")

    # Step 2: Ask Claude to fact-check its own answer
    verify_prompt = f"""Original question: {question}

Proposed answer: {initial}

Review this answer critically:
1. Are there any claims that could be wrong or unverifiable?
2. Is anything missing?
3. What is your confidence: high, medium, or low?

Return JSON: {{"verified": true/false, "confidence": "high|medium|low", "issues": [], "final_answer": "..."}}"""

    raw = ask(verify_prompt, temp=0.0)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except:
        return {"initial": initial, "raw_check": raw}

result = verified_answer(
    "What Python library is best for building RAG pipelines in 2025?",
)
print(json.dumps(result, indent=2))


# ══════════════════════════════════════════════════════
# FIX 4: CONSTRAIN THE OUTPUT SPACE
# Limit Claude to known-good answer sets
# ══════════════════════════════════════════════════════
print("\n=== Fix 4: Constrained output ===")

def safe_classify(text: str, valid_labels: list[str]) -> str:
    labels_str = ", ".join(valid_labels)
    prompt = f"""Classify the following text.
You MUST respond with exactly one of these labels: {labels_str}
No other words. No explanation. Just the label.

Text: "{text}"
Label:"""

    result = ask(prompt, temp=0.0).strip().lower()

    # Validate — if Claude hallucinates a label, catch it
    if result not in [l.lower() for l in valid_labels]:
        print(f"WARNING: unexpected label '{result}', defaulting to 'other'")
        return "other"
    return result

labels = ["bug", "feature_request", "billing", "question", "other"]
test_tickets = [
    "The export button crashes the app",
    "Can you add a dark mode?",
    "I was charged twice this month",
    "How do I invite a team member?",
]
for t in test_tickets:
    label = safe_classify(t, labels)
    print(f"  '{t[:40]}' → {label}")
