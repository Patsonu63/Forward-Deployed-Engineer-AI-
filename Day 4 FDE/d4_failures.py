import anthropic, os, json, time
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# THE 5 MOST COMMON AGENT FAILURES
# Every FDE must know these and how to prevent them
#
# 1. Infinite loops        — agent keeps calling tools endlessly
# 2. Hallucinated tool calls — agent calls tools with wrong params
# 3. Tool timeouts          — external API hangs, agent freezes
# 4. Context overflow       — message history gets too long
# 5. Runaway costs          — agent makes hundreds of API calls
# ══════════════════════════════════════════════════════


# ════════════════════════════════
# FAILURE 1: INFINITE LOOPS
# Prevention: max_steps hard limit + loop detection
# ════════════════════════════════

class LoopDetector:
    """Detect if the agent is calling the same tool with same args repeatedly."""
    def __init__(self, max_repeats: int = 3):
        self.history = []
        self.max_repeats = max_repeats

    def check(self, tool_name: str, tool_input: dict) -> bool:
        """Returns True if a loop is detected."""
        key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        self.history.append(key)
        count = self.history.count(key)
        if count >= self.max_repeats:
            print(f"LOOP DETECTED: '{tool_name}' called {count}x with same input")
            return True
        return False

# Usage in agent loop:
# detector = LoopDetector()
# for block in tool_use_blocks:
#     if detector.check(block.name, block.input):
#         return "Agent stuck in loop — stopping."


# ════════════════════════════════
# FAILURE 2: INVALID TOOL INPUTS
# Prevention: validate inputs before execution
# ════════════════════════════════

def validate_tool_input(tool_name: str, tool_input: dict, schema: dict) -> tuple[bool, str]:
    """
    Validate tool inputs against the tool's JSON schema.
    Returns (is_valid, error_message).
    """
    required = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Check required fields present
    for field in required_fields:
        if field not in tool_input:
            return False, f"Missing required field: '{field}'"

    # Check types
    for field, value in tool_input.items():
        if field not in required:
            return False, f"Unknown field: '{field}'"
        expected_type = required[field].get("type")
        if expected_type == "string" and not isinstance(value, str):
            return False, f"Field '{field}' must be string, got {type(value).__name__}"
        if expected_type == "integer" and not isinstance(value, int):
            return False, f"Field '{field}' must be integer, got {type(value).__name__}"
        # Enum validation
        if "enum" in required[field]:
            if value not in required[field]["enum"]:
                return False, f"Field '{field}' must be one of {required[field]['enum']}"

    return True, ""


# ════════════════════════════════
# FAILURE 3: TOOL TIMEOUTS
# Prevention: wrap every tool call in timeout + retry
# ════════════════════════════════

import functools, signal

def with_timeout(seconds: int):
    """Decorator: raise TimeoutError if function takes too long."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use threading for cross-platform timeout
            import threading
            result = [None]
            error  = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(timeout=seconds)

            if t.is_alive():
                raise TimeoutError(f"Tool exceeded {seconds}s timeout")
            if error[0]:
                raise error[0]
            return result[0]
        return wrapper
    return decorator


def safe_execute_tool(tool_name: str, tool_input: dict,
                       tool_schemas: dict, timeout_s: int = 8) -> str:
    """
    Production-safe tool executor with:
    - Input validation
    - Timeout protection
    - Retry on transient errors
    - Structured error responses
    """
    # 1. Validate input
    schema = tool_schemas.get(tool_name, {}).get("input_schema", {})
    valid, err = validate_tool_input(tool_name, tool_input, schema)
    if not valid:
        return json.dumps({"error": f"Invalid input: {err}", "status": "validation_error"})

    # 2. Execute with retry
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Apply timeout via threading approach
            import threading
            result = [None]
            exc = [None]

            def run():
                try:
                    from d4_tools import execute_tool  # or inline the functions
                    result[0] = execute_tool(tool_name, tool_input)
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(timeout=timeout_s)

            if t.is_alive():
                raise TimeoutError(f"Timed out after {timeout_s}s")
            if exc[0]:
                raise exc[0]

            return result[0]

        except TimeoutError:
            return json.dumps({"error": "Tool timed out", "status": "timeout"})
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.5 ** attempt)
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for {tool_name}...")
                continue
            return json.dumps({"error": str(e), "status": "error"})


# ════════════════════════════════
# FAILURE 4: CONTEXT OVERFLOW
# Prevention: sliding window on message history
# ════════════════════════════════

def trim_messages(messages: list, max_tokens: int = 150_000) -> list:
    """
    Keep message history within token budget.
    Always preserves: first user message + last N exchanges.
    Removes middle tool call/result pairs when context grows.
    """
    # Rough estimate: 1 token ≈ 4 chars
    def est_tokens(msgs):
        return sum(len(str(m)) for m in msgs) // 4

    if est_tokens(messages) <= max_tokens:
        return messages

    # Always keep first message (the task)
    first = messages[:1]
    rest  = messages[1:]

    # Remove oldest tool call + result pairs from the middle
    while est_tokens(first + rest) > max_tokens and len(rest) > 4:
        # Find and remove oldest assistant tool_use + user tool_result pair
        for i, msg in enumerate(rest):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list) and any(
                    getattr(b, "type", None) == "tool_use" or
                    (isinstance(b, dict) and b.get("type") == "tool_use")
                    for b in content
                ):
                    rest = rest[:i] + rest[i+2:]  # remove tool_use + tool_result
                    break
        else:
            break  # no more tool pairs to remove

    trimmed = first + rest
    print(f"  Context trimmed: {est_tokens(messages)} → {est_tokens(trimmed)} tokens")
    return trimmed


# ════════════════════════════════
# FAILURE 5: COST GUARDRAILS
# Prevention: budget tracking + hard stop
# ════════════════════════════════

class CostGuard:
    """Track and limit spending per agent run."""

    PRICE_INPUT  = 3.00 / 1_000_000   # per token
    PRICE_OUTPUT = 15.00 / 1_000_000  # per token

    def __init__(self, max_cost_usd: float = 0.10, max_steps: int = 15):
        self.max_cost  = max_cost_usd
        self.max_steps = max_steps
        self.total_in  = 0
        self.total_out = 0
        self.steps     = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.total_in  += input_tokens
        self.total_out += output_tokens
        self.steps     += 1

    @property
    def cost(self) -> float:
        return self.total_in * self.PRICE_INPUT + self.total_out * self.PRICE_OUTPUT

    def check(self) -> tuple[bool, str]:
        """Returns (should_stop, reason)."""
        if self.steps >= self.max_steps:
            return True, f"Max steps ({self.max_steps}) reached"
        if self.cost >= self.max_cost:
            return True, f"Cost limit (${self.max_cost:.3f}) reached at ${self.cost:.4f}"
        return False, ""

    def report(self):
        print(f"\n  [Cost Guard] Steps: {self.steps} | "
              f"Tokens: {self.total_in}in/{self.total_out}out | "
              f"Cost: ${self.cost:.5f}")


# ════════════════════════════════
# PRODUCTION-GRADE AGENT with ALL guardrails
# ════════════════════════════════

def run_safe_agent(
    user_message: str,
    tools: list,
    max_cost_usd: float = 0.05,
    max_steps: int = 10,
    verbose: bool = True
) -> dict:
    """
    Agent with all production guardrails:
    - Loop detection
    - Input validation
    - Timeout protection
    - Context trimming
    - Cost limiting
    """
    messages      = [{"role": "user", "content": user_message}]
    guard         = CostGuard(max_cost_usd, max_steps)
    loop_detector = LoopDetector(max_repeats=3)
    tool_schemas  = {t["name"]: t for t in tools}

    if verbose:
        print(f"\n{'='*60}")
        print(f"SAFE AGENT: {user_message[:80]}")
        print(f"Budget: ${max_cost_usd} | Max steps: {max_steps}")
        print(f"{'='*60}")

    while True:
        # ── Cost + step check ─────────────────────────
        should_stop, reason = guard.check()
        if should_stop:
            guard.report()
            return {"status": "stopped", "reason": reason, "answer": None}

        # ── Trim context if needed ─────────────────────
        messages = trim_messages(messages)

        # ── API call ───────────────────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        guard.record(response.usage.input_tokens, response.usage.output_tokens)

        if verbose:
            print(f"\n[Step {guard.steps}] stop={response.stop_reason} | "
                  f"cost so far: ${guard.cost:.5f}")

        # ── Done ──────────────────────────────────────
        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if b.type == "text"), "")
            guard.report()
            return {"status": "done", "answer": text, "steps": guard.steps}

        # ── Tool use ──────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Loop detection
                if loop_detector.check(block.name, block.input):
                    return {"status": "loop_detected", "answer": None}

                if verbose:
                    print(f"  TOOL: {block.name}({json.dumps(block.input)[:60]})")

                # Safe execution with validation + timeout
                result = safe_execute_tool(block.name, block.input, tool_schemas)

                if verbose:
                    print(f"  RESULT: {result[:100]}...")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result
                })

            messages.append({"role": "user", "content": tool_results})


# ── Test the safe agent ────────────────────────────────
if __name__ == "__main__":
    # Import tools from previous file
    # from d4_tools import TOOLS
    # For demo, redefine a minimal TOOLS list here

    result = run_safe_agent(
        "What is 18% GST on a ₹12,500 purchase, and what's the total after tax?",
        tools=TOOLS,  # from d4_tools.py
        max_cost_usd=0.02,
        max_steps=5
    )
    print(f"\nResult: {result['answer']}")
    print(f"Status: {result['status']} in {result.get('steps')} steps")
