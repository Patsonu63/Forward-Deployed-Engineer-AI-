import anthropic, os, json, math, requests
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# HOW TOOL USE WORKS
#
# 1. You define tools as JSON schemas (name + description + parameters)
# 2. You pass them to Claude with the API call
# 3. Claude decides WHEN and HOW to call a tool
# 4. Claude returns a tool_use block (NOT text) with name + input
# 5. YOU execute the actual function (Claude can't run code)
# 6. You return the result as a tool_result message
# 7. Claude reads the result and continues reasoning
# ══════════════════════════════════════════════════════


# ════════════════════════════════
# STEP 1: DEFINE YOUR TOOLS
# Each tool = name + description + input_schema
# Description is the most important part — Claude reads it to decide when to use the tool
# ════════════════════════════════

TOOLS = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations. Use for any arithmetic, percentages, unit conversions, or numerical reasoning. Do NOT try to calculate in your head — always use this tool for numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression. Examples: '2 + 2', '15 * 0.18', 'math.sqrt(144)', '(100 - 37) / 100'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use when the user asks about weather, temperature, or climate conditions in a specific location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Mumbai', 'London', 'New York'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Default: celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information. Use for recent events, facts you're unsure about, current prices, or anything that may have changed since your training.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-5). Default: 3",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]


# ════════════════════════════════
# STEP 2: IMPLEMENT THE ACTUAL FUNCTIONS
# Claude calls the tool — YOU run the code
# ════════════════════════════════

def calculator(expression: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        # Safe eval — only math functions allowed
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"result": result, "expression": expression, "status": "ok"}
    except Exception as e:
        return {"error": str(e), "expression": expression, "status": "error"}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Get weather using Open-Meteo (free, no API key needed).
    First geocode the city, then fetch weather.
    """
    try:
        # Step 1: geocode city → coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5
        ).json()

        if not geo.get("results"):
            return {"error": f"City '{city}' not found", "status": "error"}

        loc  = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        # Step 2: fetch current weather
        temp_unit = "celsius" if unit == "celsius" else "fahrenheit"
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "temperature_unit": temp_unit,
                "timezone": "auto"
            },
            timeout=5
        ).json()

        curr     = wx["current"]
        symbol   = "°C" if unit == "celsius" else "°F"
        wmo_desc = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                    45:"Foggy",61:"Light rain",63:"Moderate rain",80:"Rain showers",
                    95:"Thunderstorm"}
        condition = wmo_desc.get(curr["weather_code"], "Unknown")

        return {
            "city":        city,
            "temperature": f"{curr['temperature_2m']}{symbol}",
            "condition":   condition,
            "wind_speed":  f"{curr['wind_speed_10m']} km/h",
            "status":      "ok"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


def search_web(query: str, num_results: int = 3) -> dict:
    """
    Simulate web search using DuckDuckGo Instant Answer API (free).
    For production, use Serper, Brave Search, or Tavily APIs.
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=5
        ).json()

        results = []
        # Abstract (main answer)
        if resp.get("Abstract"):
            results.append({
                "title":   resp.get("Heading", query),
                "snippet": resp["Abstract"][:300],
                "source":  resp.get("AbstractSource", "DuckDuckGo")
            })
        # Related topics
        for topic in resp.get("RelatedTopics", [])[:num_results - 1]:
            if "Text" in topic:
                results.append({
                    "title":   topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "snippet": topic["Text"][:200],
                    "source":  "DuckDuckGo"
                })

        if not results:
            results = [{"snippet": f"No instant results for '{query}'. Try a more specific query.", "source": "DDG"}]

        return {"query": query, "results": results[:num_results], "status": "ok"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# ════════════════════════════════
# STEP 3: TOOL DISPATCHER
# Maps tool name → Python function
# ════════════════════════════════

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web":  search_web,
}

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result as a JSON string."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    result = fn(**tool_input)
    return json.dumps(result, indent=2)


# ════════════════════════════════
# STEP 4: SINGLE TOOL CALL — See the raw API flow
# ════════════════════════════════

def demo_single_tool_call():
    print("=== Single Tool Call (Raw API) ===\n")

    messages = [{"role": "user", "content": "What is 15% tip on a $47.50 restaurant bill?"}]

    # First API call — Claude decides to use calculator
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        tools=TOOLS,
        messages=messages
    )

    print(f"stop_reason: {response.stop_reason}")  # → "tool_use"
    print(f"content blocks: {[b.type for b in response.content]}")

    # Extract the tool_use block
    tool_block = next(b for b in response.content if b.type == "tool_use")
    print(f"\nTool called:  {tool_block.name}")
    print(f"Tool input:   {tool_block.input}")

    # Execute the actual function
    result = execute_tool(tool_block.name, tool_block.input)
    print(f"Tool result:  {result}")

    # Send result back to Claude
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type":        "tool_result",
            "tool_use_id": tool_block.id,
            "content":     result
        }]
    })

    # Second API call — Claude reads result and answers
    final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        tools=TOOLS,
        messages=messages
    )

    print(f"\nFinal answer: {final.content[0].text}")
    print(f"stop_reason:  {final.stop_reason}")   # → "end_turn"


demo_single_tool_call()
