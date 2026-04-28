import anthropic, os, json, math, time, requests
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# PRODUCTION AGENT WITH FULL EXECUTION TRACE
# This is what you'd show to an enterprise client:
# - Every thought captured
# - Every tool call logged
# - Execution timeline
# - Cost breakdown
# - JSON export for debugging
# ══════════════════════════════════════════════════════


@dataclass
class TraceEvent:
    """A single event in the agent's execution trace."""
    step:      int
    event_type: str    # "think" | "tool_call" | "tool_result" | "final_answer"
    content:   Any
    timestamp: float = field(default_factory=time.time)
    duration:  float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def cost(self) -> float:
        return self.tokens_in * 3/1_000_000 + self.tokens_out * 15/1_000_000


class TracedAgent:
    """An AI agent that captures a full execution trace for debugging and auditing."""

    def __init__(self, tools: list, system: str, max_steps: int = 15):
        self.tools     = tools
        self.system    = system
        self.max_steps = max_steps
        self.trace: list[TraceEvent] = []
        self.messages  = []
        self.step      = 0

    def _add_event(self, event_type: str, content: Any, **kwargs) -> TraceEvent:
        evt = TraceEvent(step=self.step, event_type=event_type, content=content, **kwargs)
        self.trace.append(evt)
        return evt

    def run(self, task: str) -> dict:
        """Run the agent and return result with full trace."""
        self.messages = [{"role": "user", "content": task}]
        start_time = time.time()

        self._print_header(task)

        while self.step < self.max_steps:
            self.step += 1
            t0 = time.time()

            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=self.system,
                tools=self.tools,
                messages=self.messages
            )
            api_time = time.time() - t0

            # Capture any text (thinking) blocks
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self._add_event("think", block.text,
                                    duration=api_time,
                                    tokens_in=resp.usage.input_tokens,
                                    tokens_out=resp.usage.output_tokens)
                    self._print_think(block.text)

            if resp.stop_reason == "end_turn":
                answer = next((b.text for b in resp.content if b.type == "text"), "")
                self._add_event("final_answer", answer,
                                duration=api_time,
                                tokens_in=resp.usage.input_tokens,
                                tokens_out=resp.usage.output_tokens)
                self._print_answer(answer)
                break

            if resp.stop_reason == "tool_use":
                self.messages.append({"role": "assistant", "content": resp.content})
                tool_results = []

                for block in resp.content:
                    if block.type != "tool_use":
                        continue

                    self._add_event("tool_call",
                                    {"name": block.name, "input": block.input},
                                    tokens_in=resp.usage.input_tokens,
                                    tokens_out=resp.usage.output_tokens)
                    self._print_tool_call(block.name, block.input)

                    # Execute
                    t_tool = time.time()
                    result = self._execute(block.name, block.input)
                    tool_dur = time.time() - t_tool

                    self._add_event("tool_result",
                                    {"name": block.name, "result": result},
                                    duration=tool_dur)
                    self._print_tool_result(block.name, result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

                self.messages.append({"role": "user", "content": tool_results})

        total_time = time.time() - start_time
        return self._build_report(total_time)

    def _execute(self, name: str, inp: dict) -> str:
        """Execute a tool — override this in subclasses."""
        return json.dumps({"error": f"Tool '{name}' not implemented"})

    def _build_report(self, total_time: float) -> dict:
        """Build execution report from trace."""
        total_cost   = sum(e.cost() for e in self.trace)
        total_in     = sum(e.tokens_in for e in self.trace)
        total_out    = sum(e.tokens_out for e in self.trace)
        tool_calls   = [e for e in self.trace if e.event_type == "tool_call"]
        final        = next((e for e in reversed(self.trace) if e.event_type == "final_answer"), None)

        report = {
            "answer":       final.content if final else None,
            "status":       "complete" if final else "incomplete",
            "metrics": {
                "steps":        self.step,
                "tool_calls":   len(tool_calls),
                "total_tokens": total_in + total_out,
                "input_tokens": total_in,
                "output_tokens":total_out,
                "cost_usd":     round(total_cost, 6),
                "duration_s":   round(total_time, 2),
            },
            "tool_call_summary": [
                {"tool": e.content["name"], "input": str(e.content["input"])[:60]}
                for e in tool_calls
            ],
            "trace": [
                {"step": e.step, "type": e.event_type,
                 "preview": str(e.content)[:100],
                 "tokens": e.tokens_in + e.tokens_out,
                 "cost": round(e.cost(), 6)}
                for e in self.trace
            ]
        }
        self._print_report(report)
        return report

    # ── Pretty printing ────────────────────────────────
    def _print_header(self, task):
        print(f"\n{'='*64}")
        print(f"AGENT RUN: {task[:60]}")
        print(f"{'='*64}")

    def _print_think(self, text):
        preview = text.replace('\n', ' ')[:100]
        print(f"\n[Step {self.step}] THINK: {preview}...")

    def _print_tool_call(self, name, inp):
        print(f"         CALL:  {name}({json.dumps(inp)[:70]})")

    def _print_tool_result(self, name, result):
        print(f"         RESULT:{result[:80]}...")

    def _print_answer(self, answer):
        print(f"\n{'─'*64}")
        print(f"ANSWER:\n{answer}")
        print(f"{'─'*64}")

    def _print_report(self, r):
        m = r["metrics"]
        print(f"\nMETRICS: {m['steps']} steps | {m['tool_calls']} tools | "
              f"{m['total_tokens']} tokens | ${m['cost_usd']} | {m['duration_s']}s")


# ── Concrete agent with real tools ────────────────────
class ResearchAgent(TracedAgent):

    def _execute(self, name: str, inp: dict) -> str:
        if name == "calculator":
            try:
                allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
                result = eval(inp["expression"], {"__builtins__": {}}, allowed)
                return json.dumps({"result": round(result, 4)})
            except Exception as e:
                return json.dumps({"error": str(e)})

        if name == "get_weather":
            try:
                geo = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": inp["city"], "count": 1, "format": "json"},
                    timeout=5
                ).json()
                if not geo.get("results"):
                    return json.dumps({"error": "City not found"})
                loc = geo["results"][0]
                wx = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                            "current": "temperature_2m,weather_code", "timezone": "auto"},
                    timeout=5
                ).json()
                return json.dumps({
                    "city": inp["city"],
                    "temp_c": wx["current"]["temperature_2m"],
                    "condition": wx["current"]["weather_code"]
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown tool: {name}"})


# ── Define tools for this agent ───────────────────────
RESEARCH_TOOLS = [
    {
        "name": "calculator",
        "description": "Perform math calculations. Always use for numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius","fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        }
    }
]

RESEARCH_SYSTEM = """You are a helpful assistant with math and weather tools.
Use the calculator for ALL arithmetic — never compute in your head.
Show your reasoning step by step."""


# ── Run it ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = ResearchAgent(RESEARCH_TOOLS, RESEARCH_SYSTEM)

    result = agent.run(
        "What is the weather in Bangalore? Convert the temp to Fahrenheit. "
        "Then calculate: if it's that temperature and I need it to be 22°C, "
        "how many degrees of cooling do I need?"
    )

    # Save trace to file for debugging
    with open("agent_trace.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTrace saved to agent_trace.json")
