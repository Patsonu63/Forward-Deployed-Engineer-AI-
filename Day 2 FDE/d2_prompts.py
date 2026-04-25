import anthropic, os, json
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask(prompt: str, system: str = "", temp: float = 0.3) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, temperature=temp,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip()


# ══════════════════════════════════════════════════════
# TECHNIQUE 1: ZERO-SHOT PROMPTING
# Just ask — no examples, no hints
# ══════════════════════════════════════════════════════
print("=== 1. Zero-Shot ===")
result = ask("Classify this ticket as bug, feature, or question: 'App crashes on login'")
print(result)
# Output: bug
# Works well for simple, clear tasks


# ══════════════════════════════════════════════════════
# TECHNIQUE 2: FEW-SHOT PROMPTING
# Show examples to teach the format you want
# ══════════════════════════════════════════════════════
print("\n=== 2. Few-Shot ===")
few_shot_prompt = """Classify support tickets. Reply with ONLY one word: bug, feature, or question.

Ticket: "App crashes when I click save"
Label: bug

Ticket: "Can you add dark mode?"
Label: feature

Ticket: "How do I reset my password?"
Label: question

Ticket: "Export to CSV button does nothing"
Label:"""

result = ask(few_shot_prompt)
print(result)   # → bug (consistent format, no extra text)


# ══════════════════════════════════════════════════════
# TECHNIQUE 3: CHAIN-OF-THOUGHT (CoT)
# Ask model to reason step-by-step before answering
# Use for: math, logic, complex decisions
# ══════════════════════════════════════════════════════
print("\n=== 3. Chain-of-Thought ===")
cot_prompt = """A customer is on the Pro plan ($49/month).
They've been a customer for 3 years and just submitted their 2nd support ticket this month.
Their issue is a billing error of $147 (3 months overcharged).

Think step by step, then decide: should we offer a full refund, partial credit, or escalate to billing team?

Reasoning:"""

result = ask(cot_prompt, temp=0.2)
print(result)
# Claude reasons through: tenure, error amount, ticket history → then recommends


# ══════════════════════════════════════════════════════
# TECHNIQUE 4: XML TAGS / DELIMITERS
# Structure complex prompts to avoid confusion
# Anthropic recommends XML tags for Claude
# ══════════════════════════════════════════════════════
print("\n=== 4. XML Tags / Structured Prompts ===")
customer_email = """
Hi, I've been trying to export my data for 3 days now.
The button literally does nothing when I click it.
This is urgent — I need the data for a board presentation tomorrow morning.
— Sarah
"""

structured_prompt = f"""
<task>Classify and respond to the customer support email below.</task>

<email>
{customer_email}
</email>

<instructions>
1. Identify: urgency (low/medium/high/critical), category (bug/billing/feature/question)
2. Write a professional response under 100 words
3. Suggest one internal action for the support team
</instructions>

<output_format>
Urgency: [level]
Category: [type]
Response: [your response]
Internal action: [action]
</output_format>
"""

result = ask(structured_prompt)
print(result)


# ══════════════════════════════════════════════════════
# TECHNIQUE 5: ROLE PROMPTING (System Prompt Personas)
# Give Claude a precise role, expertise level, and rules
# This is the most important technique for FDE work
# ══════════════════════════════════════════════════════
print("\n=== 5. Role Prompting via System Prompt ===")

PERSONAS = {
    "junior_dev": """You are a junior developer who just started learning Python.
Explain concepts simply. Admit when you're not sure. Use simple vocabulary.""",

    "senior_fde": """You are a senior Forward Deployed Engineer at an AI company with 8 years experience.
You give direct, opinionated answers. You prioritize production reliability.
You always mention tradeoffs. You're concise — no fluff.""",

    "sales_engineer": """You are a solutions engineer helping close enterprise deals.
Your goal is to explain technical concepts to non-technical executives.
Use analogies, avoid jargon, focus on business outcomes not implementation details."""
}

question = "What is RAG and should we use it for our customer support bot?"

for role, system in PERSONAS.items():
    print(f"\n[{role}]:")
    print(ask(question, system=system)[:200] + "...")


# ══════════════════════════════════════════════════════
# TECHNIQUE 6: STRUCTURED OUTPUT (JSON Mode)
# Force Claude to return machine-readable data
# Essential for building AI pipelines
# ══════════════════════════════════════════════════════
print("\n=== 6. Structured JSON Output ===")

def extract_ticket_data(raw_text: str) -> dict:
    system = """You are a data extraction system.
You ALWAYS respond with valid JSON only.
No explanation, no markdown, no code blocks. Raw JSON only."""

    prompt = f"""Extract structured data from this support ticket:

<ticket>{raw_text}</ticket>

Return this exact JSON structure:
{{
  "customer_name": "string or null",
  "urgency": "critical|high|medium|low",
  "category": "bug|billing|feature|question|other",
  "sentiment": "angry|frustrated|neutral|happy",
  "key_issue": "one sentence summary",
  "needs_escalation": true or false,
  "suggested_team": "billing|engineering|support|sales"
}}"""

    raw = ask(prompt, system=system, temp=0.0)

    # Clean potential markdown wrapping just in case
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Parse error: {e}\nRaw: {raw}")
        return {}


ticket = """
From: Raj Mehta <raj@enterprise.co>
Subject: URGENT - Production API down, losing $50k/hour

Our entire integration has been down for 2 hours.
Every API call returns 503. We're processing $50k/hour through this pipeline.
I need an engineer on a call NOW.
"""

data = extract_ticket_data(ticket)
print(json.dumps(data, indent=2))
# {
#   "customer_name": "Raj Mehta",
#   "urgency": "critical",
#   "category": "bug",
#   "sentiment": "angry",
#   "key_issue": "Production API returning 503 errors causing major revenue loss",
#   "needs_escalation": true,
#   "suggested_team": "engineering"
# }
