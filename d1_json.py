import json
import os

# ══════════════════════════════════════════════════════
# FILE I/O & JSON — FDEs read/write configs constantly
# ══════════════════════════════════════════════════════

# ── 1. Write JSON to a file ────────────────────────────
customer_data = {
    "customers": [
        {"id": 1, "name": "Rahul", "tier": "enterprise", "tickets": 5},
        {"id": 2, "name": "Priya", "tier": "pro", "tickets": 2},
        {"id": 3, "name": "Ankit", "tier": "free", "tickets": 0},
    ],
    "total": 3
}

with open("customers.json", "w") as f:
    json.dump(customer_data, f, indent=2)

print("Written to customers.json")


# ── 2. Read JSON from a file ───────────────────────────
with open("customers.json", "r") as f:
    loaded = json.load(f)

print(f"Total customers: {loaded['total']}")
for c in loaded["customers"]:
    print(f"  {c['name']} — {c['tier']} — {c['tickets']} tickets")


# ── 3. Parse JSON from a string (common with APIs) ────
raw_response = '{"status": "ok", "model": "claude-sonnet-4-6", "tokens": 342}'
parsed = json.loads(raw_response)   # string → dict
print(parsed["model"])              # claude-sonnet-4-6

back_to_string = json.dumps(parsed, indent=2)   # dict → string
print(back_to_string)


# ── 4. .env file & environment variables ──────────────
# Your .env file should look like this:
# ANTHROPIC_API_KEY=sk-ant-xxxxx
# OPENAI_API_KEY=sk-xxxxx
# ENVIRONMENT=development

from dotenv import load_dotenv
load_dotenv()   # reads .env file into environment

api_key = os.getenv("ANTHROPIC_API_KEY", "not-set")
env     = os.getenv("ENVIRONMENT", "development")

print(f"Environment: {env}")
print(f"API key loaded: {'yes' if api_key != 'not-set' else 'no'}")

# NEVER do this — hardcoded secret in code:
# api_key = "sk-ant-abc123"  ← BAD — never commit this

# ALWAYS do this:
# api_key = os.getenv("ANTHROPIC_API_KEY")  ← GOOD


# ── 5. Read a plain text file (e.g. prompt template) ──
# First create a sample template file
with open("prompt_template.txt", "w") as f:
    f.write("You are a helpful customer support agent for {company}.\n")
    f.write("Always respond in {language}.\n")
    f.write("Customer name: {customer_name}\n")

with open("prompt_template.txt", "r") as f:
    template = f.read()

# Fill in the template
prompt = template.format(
    company="Acme Corp",
    language="English",
    customer_name="Rahul"
)
print(prompt)


# ── 6. Handling missing files gracefully ──────────────
def load_config(filepath: str) -> dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
        return {}

config = load_config("nonexistent.json")
print(config)  # {}  — graceful fallback
