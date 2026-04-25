import anthropic, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# TOKENS — The unit of everything in LLMs
# ══════════════════════════════════════════════════════
# Rule of thumb: 1 token ≈ 4 characters ≈ 0.75 words
# "Hello world"        → ~2 tokens
# "Retrieval-Augmented" → ~4 tokens (hyphen splits it)
# 1 emoji              → 1–3 tokens
# Claude Sonnet 4.6 context window = 200,000 tokens ≈ 150,000 words ≈ 500 pages

# ── Count tokens without making a full API call ────────
def count_tokens(text: str) -> int:
    """Count tokens for a message using Anthropic's token counter."""
    response = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens

samples = [
    "Hi",
    "Hello, how are you today?",
    "Explain retrieval-augmented generation in detail.",
    "The quick brown fox jumps over the lazy dog. " * 10,
]
for s in samples:
    print(f"{len(s):4d} chars → {count_tokens(s):4d} tokens | '{s[:40]}...' ")


# ══════════════════════════════════════════════════════
# TEMPERATURE — Controls randomness
# ══════════════════════════════════════════════════════
# 0.0  = deterministic, always picks highest probability token
# 0.7  = balanced (good default for most tasks)
# 1.0  = creative, more variety
# 1.0+ = very random (use for creative writing only)

def ask_at_temp(question: str, temperature: float) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        temperature=temperature,
        messages=[{"role": "user", "content": question}]
    )
    return msg.content[0].text.strip()

question = "Give me one creative name for an AI startup."

print("\n=== Temperature Experiment ===")
for temp in [0.0, 0.5, 1.0]:
    result = ask_at_temp(question, temp)
    print(f"temp={temp}: {result}")
# temp=0.0 will give same answer every run
# temp=1.0 will give a different answer every run


# ══════════════════════════════════════════════════════
# MAX_TOKENS — Hard cap on output length
# ══════════════════════════════════════════════════════
# If the model needs 300 tokens but max_tokens=50, it cuts off mid-sentence
# Always set this generously — you pay only for what's used

def ask_with_max_tokens(q: str, max_tok: int) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tok,
        messages=[{"role": "user", "content": q}]
    )
    result = msg.content[0].text
    stop_reason = msg.stop_reason   # "end_turn" = finished | "max_tokens" = cut off
    print(f"  stop_reason: {stop_reason} | tokens used: {msg.usage.output_tokens}")
    return result

print("\n=== Max Tokens Experiment ===")
q = "Write a detailed explanation of how RAG works."
print("With max_tokens=20 (too small):")
print(ask_with_max_tokens(q, 20))
print("\nWith max_tokens=200 (comfortable):")
print(ask_with_max_tokens(q, 200))


# ══════════════════════════════════════════════════════
# TOP_P — Nucleus sampling: controls token pool size
# ══════════════════════════════════════════════════════
# top_p=1.0   → consider ALL tokens (default)
# top_p=0.9   → consider only top 90% probability mass
# top_p=0.5   → only top 50% — more focused, less variety
# Use EITHER temperature OR top_p, not both

def ask_with_top_p(q: str, top_p: float) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        top_p=top_p,
        messages=[{"role": "user", "content": q}]
    )
    return msg.content[0].text.strip()

print("\n=== Top-P Experiment ===")
q = "Complete this sentence creatively: The robot looked at the stars and..."
for p in [0.3, 0.7, 1.0]:
    result = ask_with_top_p(q, p)
    print(f"top_p={p}: {result[:80]}")


# ══════════════════════════════════════════════════════
# CONTEXT WINDOW — How much history the model can see
# ══════════════════════════════════════════════════════
# Context window = system prompt + all messages + response
# Claude Sonnet 4.6 = 200K tokens input, 8K tokens output
# When context fills up → oldest messages must be trimmed

def trim_history(messages: list, max_tokens: int = 180000) -> list:
    """
    Trim oldest messages if context gets too large.
    Always keep the most recent exchanges.
    """
    while True:
        total = count_tokens(str(messages))
        if total <= max_tokens or len(messages) <= 2:
            break
        messages.pop(0)  # remove oldest user message
        if messages:
            messages.pop(0)  # remove corresponding assistant message
    return messages

# Simulate a long conversation
messages = []
for i in range(3):
    messages.append({"role": "user", "content": f"Question {i+1}: What is token {i+1}?"})
    messages.append({"role": "assistant", "content": f"Token {i+1} is a unit of text processed by the LLM."})

print(f"\nMessages before trim: {len(messages)}")
messages = trim_history(messages, max_tokens=50)
print(f"Messages after trim: {len(messages)}")
