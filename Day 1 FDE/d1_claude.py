import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
# YOUR FIRST CLAUDE API CALL
# Get your API key from: console.anthropic.com
# Add to .env: ANTHROPIC_API_KEY=sk-ant-xxxxx
# ══════════════════════════════════════════════════════

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── 1. Simplest possible call ─────────────────────────
def ask_claude(question: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return message.content[0].text

response = ask_claude("What is RAG in AI? Explain in 2 sentences.")
print(response)


# ── 2. System prompt — give Claude a persona ──────────
def support_agent(user_message: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="""You are a senior technical support engineer at an AI software company.
You respond concisely and professionally.
Always ask one clarifying question at the end if the issue is unclear.
Never make up information — say 'I don't know' if unsure.""",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return message.content[0].text

reply = support_agent("Our API keeps returning 500 errors when we send large payloads.")
print(reply)


# ── 3. Multi-turn conversation (memory via message list) ──
def chat_session():
    messages = []   # this list is our "memory"

    print("Chat with Claude (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="You are a helpful FDE mentor teaching AI engineering concepts.",
            messages=messages   # pass full history each time
        )

        assistant_reply = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_reply})
        print(f"\nClaude: {assistant_reply}\n")


# ── 4. Structured output — ask Claude to return JSON ──
import json

def classify_support_ticket(ticket: str) -> dict:
    prompt = f"""Classify this support ticket and return ONLY valid JSON.
No explanation, no markdown, just raw JSON.

Ticket: "{ticket}"

Return this exact structure:
{{
  "category": "one of: auth, api, billing, performance, integration, other",
  "priority": "one of: critical, high, medium, low",
  "summary": "one sentence summary",
  "suggested_action": "one sentence recommended next step"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Claude returned non-JSON: {raw}")
        return {}

ticket_text = "Users cannot log in after we enabled SSO with Okta. Auth fails silently."
result = classify_support_ticket(ticket_text)
print(json.dumps(result, indent=2))
# {
#   "category": "auth",
#   "priority": "critical",
#   "summary": "SSO authentication failure after Okta integration",
#   "suggested_action": "Check SAML assertion attributes and Okta app config"
# }


# ── 5. Token usage & cost awareness ──────────────────
def ask_with_usage(question: str) -> tuple[str, dict]:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": question}]
    )
    usage = {
        "input_tokens":  message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens":  message.usage.input_tokens + message.usage.output_tokens
    }
    return message.content[0].text, usage

answer, usage = ask_with_usage("What is LangChain in one sentence?")
print(f"Answer: {answer}")
print(f"Tokens used: {usage}")
# FDEs must track token usage to manage enterprise client costs


# ── Day 1 capstone: Put it all together ───────────────
def fde_day1_demo():
    print("=== FDE Day 1 Demo ===\n")

    # 1. Fetch a GitHub repo via REST API
    response = __import__("requests").get("https://api.github.com/repos/anthropics/anthropic-sdk-python")
    repo = response.json()
    print(f"Repo: {repo['full_name']}")
    print(f"Stars: {repo['stargazers_count']:,}")
    print(f"Description: {repo['description']}\n")

    # 2. Ask Claude to summarize it
    summary_prompt = f"""Given this GitHub repo info, write a one-line description for a non-technical stakeholder:
Name: {repo['full_name']}
Description: {repo['description']}
Stars: {repo['stargazers_count']}"""

    summary = ask_claude(summary_prompt)
    print(f"Claude summary: {summary}")

if __name__ == "__main__":
    fde_day1_demo()
