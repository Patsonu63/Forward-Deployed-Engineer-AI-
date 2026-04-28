import anthropic, os, json, math, time, requests
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Paste your TOOLS list and TOOL_FUNCTIONS dict from d4_tools.py here ──
# (or import them: from d4_tools import TOOLS, execute_tool)

# ... [assume TOOLS, execute_tool defined above] ...

# ══════════════════════════════════════════════════════
# THE REACT AGENT LOOP
# This is the core pattern you'll use for every agent
#
# while True:
#   call Claude with messages + tools
#   if stop_reason == "end_turn":   → Claude is done, return answer
#   if stop_reason == "tool_use":   → execute tools, add results, loop
#   if max_steps reached:           → safety exit
# ══════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

Always follow this approach:
1. Break complex questions into sub-tasks
2. Use the calculator tool for ALL math — never calculate in your head
3. Use search_web for any facts you're not 100% certain about
4. Use get_weather only when explicitly asked about weather
5. Think step by step before acting
6. Be concise in your final answer"""


def run_agent(
    user_message: str,
    tools: list,
    max_steps: int = 10,
    verbose: bool = True
) -> str:
    """
    Full ReAct agent loop.
    Returns the final answer string.
    """
    messages  = [{"role": "user", "content": user_message}]
    step      = 0
    total_in  = 0
    total_out = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_message}")
        print(f"{'='*60}")

    while step < max_steps:
        step += 1
        if verbose:
            print(f"\n[Step {step}] Calling Claude...")

        # ── Call Claude ────────────────────────────────
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        total_in  += response.usage.input_tokens
        total_out += response.usage.output_tokens

        if verbose:
            print(f"  stop_reason: {response.stop_reason}")
            print(f"  content:     {[b.type for b in response.content]}")

        # ── Case 1: Claude is done ─────────────────────
        if response.stop_reason == "end_turn":
            text_block = next((b for b in response.content if b.type == "text"), None)
            answer = text_block.text if text_block else "No answer generated."
            if verbose:
                print(f"\n{'─'*60}")
                print(f"FINAL ANSWER:\n{answer}")
                print(f"{'─'*60}")
                print(f"Steps: {step} | Tokens: {total_in}in/{total_out}out")
            return answer

        # ── Case 2: Claude wants to use tools ──────────
        if response.stop_reason == "tool_use":
            # Add Claude's response to message history
            messages.append({"role": "assistant", "content": response.content})

            # Process ALL tool calls in this response (Claude may call multiple)
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if verbose:
                    print(f"\n  TOOL CALL: {block.name}")
                    print(f"  INPUT:     {json.dumps(block.input, indent=4)}")

                # Execute the tool
                t_start = time.time()
                result  = execute_tool(block.name, block.input)
                t_end   = time.time()

                if verbose:
                    result_preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"  RESULT:    {result_preview}")
                    print(f"  TIME:      {t_end-t_start:.2f}s")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result
                })

            # Add all tool results as a single user message
            messages.append({"role": "user", "content": tool_results})
            continue

        # ── Case 3: Unexpected stop reason ────────────
        print(f"WARNING: Unexpected stop_reason: {response.stop_reason}")
        break

    return "Agent exceeded maximum steps without completing the task."


# ════════════════════════════════
# TEST THE AGENT
# ════════════════════════════════

# Test 1: Pure math — should use calculator
print("\n" + "="*60)
print("TEST 1: Multi-step math")
run_agent(
    "If I invest ₹50,000 at 8.5% annual interest for 3 years (compound), how much will I have? Show the calculation.",
    tools=TOOLS
)

# Test 2: Multi-tool — weather + math
print("\n" + "="*60)
print("TEST 2: Weather + unit conversion")
run_agent(
    "What is the weather in Mumbai right now? Also convert that temperature to Fahrenheit.",
    tools=TOOLS
)

# Test 3: Research question — should use search
print("\n" + "="*60)
print("TEST 3: Research + calculation")
run_agent(
    "What is LangChain used for? Then calculate: if it has 85,000 GitHub stars and grows at 2% per month, how many stars in 6 months?",
    tools=TOOLS
)
