import anthropic, os, time
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# FULL API ANATOMY — Every field you'll use as an FDE
# ══════════════════════════════════════════════════════

response = client.messages.create(
    # ── Required ──────────────────────────────────────
    model      = "claude-sonnet-4-6",  # model to use
    max_tokens = 1024,                 # hard output cap (always set this)

    # ── Optional but important ────────────────────────
    system     = "You are a concise technical assistant.",  # persona + rules
    temperature= 0.3,    # 0=deterministic, 1=creative
    top_p      = 1.0,    # nucleus sampling (use either temp OR top_p)

    # ── The conversation ──────────────────────────────
    messages   = [
        {"role": "user",      "content": "What is an embedding?"},
        {"role": "assistant", "content": "An embedding is a dense vector representation of text."},
        {"role": "user",      "content": "Give me a one-line Python example."},
    ]
    # The messages list IS your memory — Claude has no state between calls
)

# ── Reading the response object ───────────────────────
print("=== Response Object Anatomy ===")
print(f"id:           {response.id}")
print(f"model:        {response.model}")
print(f"stop_reason:  {response.stop_reason}")  # end_turn | max_tokens | stop_sequence
print(f"input_tokens: {response.usage.input_tokens}")
print(f"output_tokens:{response.usage.output_tokens}")
print(f"content type: {response.content[0].type}")   # "text"
print(f"text:         {response.content[0].text}")


# ══════════════════════════════════════════════════════
# STOP SEQUENCES — Tell Claude when to stop
# ══════════════════════════════════════════════════════
# Useful when you want Claude to stop at a delimiter
# e.g., stop after generating a JSON block

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=300,
    stop_sequences=["```", "END"],   # stop when Claude hits ``` or END
    messages=[{
        "role": "user",
        "content": "Write a Python function to add two numbers. Show just the code, no explanation. Start with ```python"
    }]
)
print(f"\nstop_reason: {response.stop_reason}")  # "stop_sequence"
print(response.content[0].text)


# ══════════════════════════════════════════════════════
# STREAMING — Get tokens as they arrive (better UX)
# ══════════════════════════════════════════════════════
# Without streaming: wait 3-10 seconds, then get full response
# With streaming: tokens appear immediately like in Claude.ai

print("\n=== Streaming Demo ===")
print("Claude: ", end="", flush=True)

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}]
) as stream:
    for text_chunk in stream.text_stream:
        print(text_chunk, end="", flush=True)
        # In a web app, you'd send each chunk to the frontend via SSE or WebSocket

print()  # newline after stream


# ══════════════════════════════════════════════════════
# COST CALCULATOR — FDEs must track spend per client
# ══════════════════════════════════════════════════════
# Claude Sonnet 4.6 pricing (as of 2025):
# Input:  $3.00 per 1M tokens
# Output: $15.00 per 1M tokens

PRICE_INPUT  = 3.00 / 1_000_000   # per token
PRICE_OUTPUT = 15.00 / 1_000_000  # per token

def calculate_cost(input_tokens: int, output_tokens: int) -> dict:
    input_cost  = input_tokens  * PRICE_INPUT
    output_cost = output_tokens * PRICE_OUTPUT
    total       = input_cost + output_cost
    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "input_cost":    f"${input_cost:.6f}",
        "output_cost":   f"${output_cost:.6f}",
        "total_cost":    f"${total:.6f}",
        "monthly_1000_calls": f"${total * 1000:.2f}"
    }

# Simulate a typical enterprise call
result = calculate_cost(input_tokens=2500, output_tokens=400)
for k, v in result.items():
    print(f"  {k}: {v}")

# ── Tracked API call wrapper ──────────────────────────
class TrackedClient:
    """Wraps the Anthropic client with cost + latency tracking."""
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.total_input  = 0
        self.total_output = 0
        self.call_count   = 0

    def ask(self, messages: list, system: str = "", max_tokens: int = 512) -> str:
        start = time.time()
        resp  = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=messages
        )
        elapsed = time.time() - start

        self.total_input  += resp.usage.input_tokens
        self.total_output += resp.usage.output_tokens
        self.call_count   += 1

        print(f"[call #{self.call_count}] {resp.usage.input_tokens}in/{resp.usage.output_tokens}out | {elapsed:.2f}s")
        return resp.content[0].text

    def report(self):
        cost = calculate_cost(self.total_input, self.total_output)
        print(f"\n=== Session Report ({self.call_count} calls) ===")
        print(f"  Total tokens: {self.total_input + self.total_output:,}")
        print(f"  Total cost:   {cost['total_cost']}")


tracked = TrackedClient()
tracked.ask([{"role": "user", "content": "What is LangChain?"}])
tracked.ask([{"role": "user", "content": "What is LangGraph?"}])
tracked.report()
