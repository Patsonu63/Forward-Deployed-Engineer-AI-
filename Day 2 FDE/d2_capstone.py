import anthropic, os, json, time
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════
# DAY 2 CAPSTONE: Enterprise Support Triage System
# Combines: system prompts, few-shot, structured output,
#           grounding, hallucination prevention, cost tracking
# This is the kind of thing you'd build in week 1 as an FDE
# ══════════════════════════════════════════════════════

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Knowledge base (in production, this comes from RAG) ─
COMPANY_POLICIES = """
SLA Policy:
- Critical (P0): 1 hour response, engineering page-out required
- High (P1): 4 hour response, senior support assigned
- Medium (P2): 24 hour response, standard support queue
- Low (P3): 72 hour response, self-service first

Escalation Rules:
- Revenue impact > $10k/hour → P0 automatically
- Data loss or breach → P0 automatically
- 3+ tickets from same customer this week → escalate to CSM
- Enterprise tier customers: always P1 minimum

Refund Policy:
- < 30 days: full refund
- 30-90 days: prorated credit
- > 90 days: goodwill credit up to $200, escalate to billing
"""

SYSTEM_PROMPT = f"""You are Triage-AI, an enterprise support ticket classifier for a B2B SaaS company.

You have access to these company policies:
<policies>
{COMPANY_POLICIES}
</policies>

Rules:
1. Apply policies EXACTLY — do not invent rules
2. If you're uncertain, set confidence to "low" and flag for human review
3. Never make up customer details
4. Return ONLY valid JSON — no markdown, no explanation

Your output must always be this exact JSON structure:
{{
  "priority": "P0|P1|P2|P3",
  "category": "bug|billing|question|feature|security|other",
  "sentiment": "critical|frustrated|neutral|satisfied",
  "summary": "one sentence",
  "sla_hours": number,
  "auto_response": "string under 80 words",
  "internal_notes": "string for support team",
  "escalate_to": "engineering|billing|csm|none",
  "confidence": "high|medium|low",
  "policy_applied": "which policy rule triggered this decision"
}}"""


def triage_ticket(ticket_text: str, customer_tier: str = "standard") -> dict:
    """
    Triage a support ticket using Claude with full policy grounding.
    Returns structured data for the support system.
    """
    start = time.time()

    prompt = f"""<customer_tier>{customer_tier}</customer_tier>

<ticket>
{ticket_text}
</ticket>

Classify this ticket per company policies. Return JSON only."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        temperature=0.0,       # deterministic — we want consistent classification
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    elapsed = round(time.time() - start, 2)
    raw = response.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw)
        result["_meta"] = {
            "latency_s": elapsed,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": round(
                response.usage.input_tokens  * 3.00/1_000_000 +
                response.usage.output_tokens * 15.00/1_000_000, 6
            )
        }
        return result
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": raw, "_meta": {"latency_s": elapsed}}


def print_triage(ticket: str, tier: str = "standard"):
    print(f"\n{'='*60}")
    print(f"TICKET (tier={tier}):\n{ticket.strip()}")
    print(f"{'─'*60}")
    result = triage_ticket(ticket, tier)

    if "error" in result:
        print(f"ERROR: {result}")
        return

    meta = result.pop("_meta")
    print(f"Priority:       {result.get('priority')} (SLA: {result.get('sla_hours')}h)")
    print(f"Category:       {result.get('category')}")
    print(f"Sentiment:      {result.get('sentiment')}")
    print(f"Escalate to:    {result.get('escalate_to')}")
    print(f"Confidence:     {result.get('confidence')}")
    print(f"Policy applied: {result.get('policy_applied')}")
    print(f"\nSummary: {result.get('summary')}")
    print(f"\nAuto-response:\n  {result.get('auto_response')}")
    print(f"\nInternal notes:\n  {result.get('internal_notes')}")
    print(f"\n[{meta['latency_s']}s | {meta['input_tokens']}in/{meta['output_tokens']}out | ${meta['cost_usd']}]")


# ── Test with real-world tickets ──────────────────────
tickets = [
    ("""Our production environment has been completely down for 90 minutes.
Every API call returns 500. We're an e-commerce platform and losing roughly
$80,000 per hour in sales. I need an engineer on the phone NOW.""", "enterprise"),

    ("""Hi, I was charged $299 last month but I'm on the $99/month plan.
Can someone look into this? No rush, just want it sorted out.
Thanks! - Meera""", "standard"),

    ("""Getting a weird error: 'TypeError: Cannot read property of undefined'
when I click the export button. Happens every time. Chrome 124, MacOS.
Screenshot attached.""", "pro"),

    ("""Hey, would love to see dark mode added to the dashboard!
Just a small thing but would make it nicer to use at night :)""", "standard"),
]

for ticket_text, tier in tickets:
    print_triage(ticket_text, tier)
